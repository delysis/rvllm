//! Working-set-derived resource budgets for Apple inference.
//!
//! We keep weights, the three in-flight scratch slots, metadata, and paged KV
//! as independent accounts.  KV receives only the remainder after fixed
//! resources and the platform safety reserve have been charged.

pub const IN_FLIGHT_SCRATCH_SLOTS: usize = 3;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AppleMemoryPlatform {
    MacOs,
    Ios,
}

impl AppleMemoryPlatform {
    #[must_use]
    pub const fn default_reserve_percent(self) -> u8 {
        match self {
            Self::MacOs => 20,
            Self::Ios => 30,
        }
    }

    #[must_use]
    pub const fn current() -> Option<Self> {
        if cfg!(target_os = "macos") {
            Some(Self::MacOs)
        } else if cfg!(target_os = "ios") {
            Some(Self::Ios)
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AppleMemoryBudgetInput {
    pub recommended_max_working_set_size: u64,
    pub weights_bytes: u64,
    pub scratch_slot_bytes: u64,
    pub metadata_bytes: u64,
    pub reserve_percent: u8,
}

impl AppleMemoryBudgetInput {
    #[must_use]
    pub const fn with_platform_defaults(
        platform: AppleMemoryPlatform,
        recommended_max_working_set_size: u64,
        weights_bytes: u64,
        scratch_slot_bytes: u64,
        metadata_bytes: u64,
    ) -> Self {
        Self {
            recommended_max_working_set_size,
            weights_bytes,
            scratch_slot_bytes,
            metadata_bytes,
            reserve_percent: platform.default_reserve_percent(),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AppleMemoryBudget {
    pub working_set_bytes: u64,
    pub reserve_bytes: u64,
    pub usable_bytes: u64,
    pub weights_bytes: u64,
    pub scratch_bytes: u64,
    pub metadata_bytes: u64,
    pub kv_pool_bytes: u64,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AppleMemoryBudgetError {
    MissingWorkingSetLimit,
    InvalidReservePercent,
    ArithmeticOverflow,
    FixedResourcesExceedBudget { required: u64, usable: u64 },
    ZeroSizedKvPage,
    ZeroUsefulKvPages,
    InsufficientKvPages { available: u64, required: u64 },
    ArithmeticNarrowing,
}

/// Physical page selection after applying the live working-set budget.
///
/// `max_useful_pages` is the largest pool the configured model shape could
/// address. `min_required_pages` expresses the admission contract which must
/// remain possible after budgeting (for example one complete max-context
/// request, or all explicitly configured max-context concurrent requests).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AppleKvPageCapacity {
    pub budget: AppleMemoryBudget,
    pub bytes_per_page: u64,
    pub max_useful_pages: u64,
    pub physical_pages: u64,
    pub allocated_kv_bytes: u64,
}

impl AppleMemoryBudget {
    pub fn derive(input: AppleMemoryBudgetInput) -> Result<Self, AppleMemoryBudgetError> {
        if input.recommended_max_working_set_size == 0 {
            return Err(AppleMemoryBudgetError::MissingWorkingSetLimit);
        }
        if input.reserve_percent >= 100 {
            return Err(AppleMemoryBudgetError::InvalidReservePercent);
        }

        let reserve_bytes = input
            .recommended_max_working_set_size
            .checked_mul(u64::from(input.reserve_percent))
            .ok_or(AppleMemoryBudgetError::ArithmeticOverflow)?
            / 100;
        let usable_bytes = input
            .recommended_max_working_set_size
            .checked_sub(reserve_bytes)
            .ok_or(AppleMemoryBudgetError::ArithmeticOverflow)?;
        let scratch_bytes = input
            .scratch_slot_bytes
            .checked_mul(IN_FLIGHT_SCRATCH_SLOTS as u64)
            .ok_or(AppleMemoryBudgetError::ArithmeticOverflow)?;
        let fixed_bytes = input
            .weights_bytes
            .checked_add(scratch_bytes)
            .and_then(|bytes| bytes.checked_add(input.metadata_bytes))
            .ok_or(AppleMemoryBudgetError::ArithmeticOverflow)?;
        let kv_pool_bytes = usable_bytes.checked_sub(fixed_bytes).ok_or(
            AppleMemoryBudgetError::FixedResourcesExceedBudget {
                required: fixed_bytes,
                usable: usable_bytes,
            },
        )?;

        Ok(Self {
            working_set_bytes: input.recommended_max_working_set_size,
            reserve_bytes,
            usable_bytes,
            weights_bytes: input.weights_bytes,
            scratch_bytes,
            metadata_bytes: input.metadata_bytes,
            kv_pool_bytes,
        })
    }

    /// Number of complete physical KV pages which fit after all fixed accounts.
    pub fn kv_page_capacity(&self, bytes_per_page: u64) -> Result<u64, AppleMemoryBudgetError> {
        if bytes_per_page == 0 {
            return Err(AppleMemoryBudgetError::ZeroSizedKvPage);
        }
        Ok(self.kv_pool_bytes / bytes_per_page)
    }

    /// Select a useful, whole-page KV pool and reject configurations whose
    /// stated admission contract cannot fit inside the device budget.
    pub fn plan_kv_pages(
        self,
        bytes_per_page: u64,
        max_useful_pages: u64,
        min_required_pages: u64,
    ) -> Result<AppleKvPageCapacity, AppleMemoryBudgetError> {
        if max_useful_pages == 0 {
            return Err(AppleMemoryBudgetError::ZeroUsefulKvPages);
        }
        let budget_pages = self.kv_page_capacity(bytes_per_page)?;
        let physical_pages = budget_pages.min(max_useful_pages);
        if physical_pages < min_required_pages {
            return Err(AppleMemoryBudgetError::InsufficientKvPages {
                available: physical_pages,
                required: min_required_pages,
            });
        }
        let allocated_kv_bytes = physical_pages
            .checked_mul(bytes_per_page)
            .ok_or(AppleMemoryBudgetError::ArithmeticOverflow)?;
        Ok(AppleKvPageCapacity {
            budget: self,
            bytes_per_page,
            max_useful_pages,
            physical_pages,
            allocated_kv_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_defaults_keep_the_documented_reserve() {
        assert_eq!(AppleMemoryPlatform::MacOs.default_reserve_percent(), 20);
        assert_eq!(AppleMemoryPlatform::Ios.default_reserve_percent(), 30);
    }

    #[test]
    fn kv_gets_only_the_remainder_after_three_scratch_slots() {
        let budget = AppleMemoryBudget::derive(AppleMemoryBudgetInput::with_platform_defaults(
            AppleMemoryPlatform::MacOs,
            1_000,
            300,
            100,
            50,
        ))
        .expect("budget");
        assert_eq!(budget.reserve_bytes, 200);
        assert_eq!(budget.usable_bytes, 800);
        assert_eq!(budget.scratch_bytes, 300);
        assert_eq!(budget.kv_pool_bytes, 150);
        assert_eq!(budget.kv_page_capacity(32).unwrap(), 4);
    }

    #[test]
    fn fixed_resources_fail_closed_instead_of_overcommitting() {
        let error = AppleMemoryBudget::derive(AppleMemoryBudgetInput::with_platform_defaults(
            AppleMemoryPlatform::Ios,
            1_000,
            600,
            100,
            1,
        ))
        .unwrap_err();
        assert_eq!(
            error,
            AppleMemoryBudgetError::FixedResourcesExceedBudget {
                required: 901,
                usable: 700,
            }
        );
    }

    #[test]
    fn zero_or_overflowing_inputs_fail_closed() {
        let mut input =
            AppleMemoryBudgetInput::with_platform_defaults(AppleMemoryPlatform::MacOs, 0, 0, 0, 0);
        assert_eq!(
            AppleMemoryBudget::derive(input),
            Err(AppleMemoryBudgetError::MissingWorkingSetLimit)
        );
        input.recommended_max_working_set_size = u64::MAX;
        input.scratch_slot_bytes = u64::MAX;
        assert_eq!(
            AppleMemoryBudget::derive(input),
            Err(AppleMemoryBudgetError::ArithmeticOverflow)
        );
    }

    #[test]
    fn page_capacity_clamps_to_the_useful_model_shape() {
        let budget = AppleMemoryBudget::derive(AppleMemoryBudgetInput {
            recommended_max_working_set_size: 10_000,
            weights_bytes: 1_000,
            scratch_slot_bytes: 100,
            metadata_bytes: 100,
            reserve_percent: 20,
        })
        .unwrap();
        let plan = budget.plan_kv_pages(32, 64, 8).unwrap();
        assert_eq!(plan.physical_pages, 64);
        assert_eq!(plan.allocated_kv_bytes, 2_048);
        assert!(plan.budget.kv_pool_bytes > plan.allocated_kv_bytes);
    }

    #[test]
    fn page_capacity_uses_only_complete_pages() {
        let budget = AppleMemoryBudget::derive(AppleMemoryBudgetInput {
            recommended_max_working_set_size: 1_000,
            weights_bytes: 500,
            scratch_slot_bytes: 0,
            metadata_bytes: 0,
            reserve_percent: 0,
        })
        .unwrap();
        let plan = budget.plan_kv_pages(128, 100, 3).unwrap();
        assert_eq!(plan.physical_pages, 3);
        assert_eq!(plan.allocated_kv_bytes, 384);
    }

    #[test]
    fn page_capacity_fails_closed_below_admission_contract() {
        let budget = AppleMemoryBudget::derive(AppleMemoryBudgetInput {
            recommended_max_working_set_size: 1_000,
            weights_bytes: 500,
            scratch_slot_bytes: 0,
            metadata_bytes: 0,
            reserve_percent: 0,
        })
        .unwrap();
        assert_eq!(
            budget.plan_kv_pages(128, 100, 4),
            Err(AppleMemoryBudgetError::InsufficientKvPages {
                available: 3,
                required: 4,
            })
        );
    }
}
