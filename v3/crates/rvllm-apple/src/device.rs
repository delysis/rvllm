use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AppleGpuFamily {
    Unknown,
    Apple7,
    Apple8,
    Apple9,
    Apple10,
}

impl AppleGpuFamily {
    #[must_use]
    pub const fn architecture_gen(self) -> u32 {
        match self {
            AppleGpuFamily::Apple7 => 14,
            AppleGpuFamily::Apple8 => 15,
            AppleGpuFamily::Apple9 => 16,
            AppleGpuFamily::Apple10 => 17,
            AppleGpuFamily::Unknown => 0,
        }
    }

    #[must_use]
    pub const fn has_nax(self) -> bool {
        matches!(self, AppleGpuFamily::Apple10)
    }

    #[must_use]
    pub const fn is_supported(self) -> bool {
        !matches!(self, AppleGpuFamily::Unknown)
    }

    #[must_use]
    pub const fn ane_cores_per_die(self) -> u32 {
        match self {
            AppleGpuFamily::Unknown => 0,
            AppleGpuFamily::Apple7
            | AppleGpuFamily::Apple8
            | AppleGpuFamily::Apple9
            | AppleGpuFamily::Apple10 => 16,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum DeviceTier {
    Base,
    Pro,
    Max,
    Ultra,
}

impl DeviceTier {
    #[must_use]
    pub const fn batch_multiplier(self) -> u32 {
        match self {
            DeviceTier::Base => 1,
            DeviceTier::Pro => 2,
            DeviceTier::Max => 4,
            DeviceTier::Ultra => 8,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum AppleNpuGeneration {
    Unknown,
    M1,
    M2,
    M3,
    M4,
    M5,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AppleAcceleratorTarget {
    pub device_name: String,
    pub gpu_family: AppleGpuFamily,
    pub tier: DeviceTier,
    pub npu_generation: AppleNpuGeneration,
    pub architecture_gen: u32,
    pub has_nax: bool,
    pub ane_cores: u32,
    pub die_count: u32,
}

impl AppleAcceleratorTarget {
    #[must_use]
    pub fn from_device_name(name: &str, die_count: u32) -> Self {
        let gpu_family = detect_gpu_family(name);
        let tier = detect_device_tier(name);
        let npu_generation = detect_npu_generation(name);
        let die_count = die_count.max(1);
        let ane_dies = match tier {
            DeviceTier::Ultra => die_count.max(2),
            _ => die_count,
        };
        let ane_cores = gpu_family.ane_cores_per_die() * ane_dies;
        Self {
            device_name: name.to_owned(),
            gpu_family,
            tier,
            npu_generation,
            architecture_gen: gpu_family.architecture_gen(),
            has_nax: gpu_family.has_nax(),
            ane_cores,
            die_count,
        }
    }

    #[must_use]
    pub const fn recommended_tile_size(&self) -> (u32, u32, u32) {
        if !self.gpu_family.is_supported() {
            (0, 0, 0)
        } else if self.has_nax {
            match self.tier {
                DeviceTier::Ultra | DeviceTier::Max => (128, 64, 32),
                DeviceTier::Pro => (64, 64, 32),
                DeviceTier::Base => (64, 32, 32),
            }
        } else {
            match self.tier {
                DeviceTier::Ultra | DeviceTier::Max => (64, 64, 32),
                DeviceTier::Pro => (64, 32, 32),
                DeviceTier::Base => (32, 32, 32),
            }
        }
    }

    #[must_use]
    pub fn cache_key(&self) -> String {
        format!(
            "{}:{:?}:{:?}:{:?}:dies{}",
            self.device_name, self.gpu_family, self.tier, self.npu_generation, self.die_count
        )
    }
}

#[must_use]
pub fn detect_gpu_family(name: &str) -> AppleGpuFamily {
    if has_chip_id(name, "M5") {
        AppleGpuFamily::Apple10
    } else if has_chip_id(name, "M4") || has_chip_id(name, "M3") || has_chip_id(name, "A17") {
        AppleGpuFamily::Apple9
    } else if has_chip_id(name, "M2") || has_chip_id(name, "A16") || has_chip_id(name, "A15") {
        AppleGpuFamily::Apple8
    } else if has_chip_id(name, "M1") || has_chip_id(name, "A14") {
        AppleGpuFamily::Apple7
    } else {
        AppleGpuFamily::Unknown
    }
}

#[must_use]
pub fn detect_device_tier(name: &str) -> DeviceTier {
    if name.contains("Ultra") {
        DeviceTier::Ultra
    } else if name.contains("Max") {
        DeviceTier::Max
    } else if name.contains("Pro") {
        DeviceTier::Pro
    } else {
        DeviceTier::Base
    }
}

#[must_use]
pub fn detect_npu_generation(name: &str) -> AppleNpuGeneration {
    if has_chip_id(name, "M5") {
        AppleNpuGeneration::M5
    } else if has_chip_id(name, "M4") {
        AppleNpuGeneration::M4
    } else if has_chip_id(name, "M3") {
        AppleNpuGeneration::M3
    } else if has_chip_id(name, "M2") {
        AppleNpuGeneration::M2
    } else if has_chip_id(name, "M1") {
        AppleNpuGeneration::M1
    } else {
        AppleNpuGeneration::Unknown
    }
}

#[must_use]
fn has_chip_id(name: &str, chip_id: &str) -> bool {
    let mut start = 0;
    while let Some(pos) = name[start..].find(chip_id) {
        let abs = start + pos;
        let after = abs + chip_id.len();
        if after >= name.len() || !name.as_bytes()[after].is_ascii_digit() {
            return true;
        }
        start = after;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_policy_covers_apple7_through_apple10_and_tiers() {
        let cases = [
            (
                "Apple M1",
                1,
                AppleGpuFamily::Apple7,
                DeviceTier::Base,
                false,
                16,
                (32, 32, 32),
            ),
            (
                "Apple M2 Pro",
                1,
                AppleGpuFamily::Apple8,
                DeviceTier::Pro,
                false,
                16,
                (64, 32, 32),
            ),
            (
                "Apple M3 Max",
                1,
                AppleGpuFamily::Apple9,
                DeviceTier::Max,
                false,
                16,
                (64, 64, 32),
            ),
            (
                "Apple M5 Ultra",
                2,
                AppleGpuFamily::Apple10,
                DeviceTier::Ultra,
                true,
                32,
                (128, 64, 32),
            ),
        ];

        for (name, die_count, family, tier, has_nax, ane_cores, tile_size) in cases {
            let target = AppleAcceleratorTarget::from_device_name(name, die_count);
            assert_eq!(target.gpu_family, family, "{name}");
            assert_eq!(target.tier, tier, "{name}");
            assert_eq!(target.architecture_gen, family.architecture_gen(), "{name}");
            assert_eq!(target.has_nax, has_nax, "{name}");
            assert_eq!(target.ane_cores, ane_cores, "{name}");
            assert_eq!(target.recommended_tile_size(), tile_size, "{name}");
        }
    }

    #[test]
    fn m10_target_does_not_receive_implicit_accelerator_policy() {
        let target = AppleAcceleratorTarget::from_device_name("Apple M10 Max", 2);
        assert_eq!(target.gpu_family, AppleGpuFamily::Unknown);
        assert_eq!(target.npu_generation, AppleNpuGeneration::Unknown);
        assert_eq!(target.architecture_gen, 0);
        assert!(!target.has_nax);
        assert_eq!(target.ane_cores, 0);
        assert_eq!(target.recommended_tile_size(), (0, 0, 0));
    }

    #[test]
    fn detects_family_tier_nax_and_ane_cores() {
        let t = AppleAcceleratorTarget::from_device_name("Apple M5 Pro", 1);
        assert_eq!(t.gpu_family, AppleGpuFamily::Apple10);
        assert_eq!(t.tier, DeviceTier::Pro);
        assert!(t.has_nax);
        assert_eq!(t.ane_cores, 16);
        assert_eq!(t.recommended_tile_size(), (64, 64, 32));
    }

    #[test]
    fn m10_does_not_match_m1() {
        assert_eq!(detect_gpu_family("Apple M10 Max"), AppleGpuFamily::Unknown);
        assert_eq!(
            detect_npu_generation("Apple M10 Max"),
            AppleNpuGeneration::Unknown
        );
    }

    #[test]
    fn ultra_has_two_die_ane_core_estimate() {
        let t = AppleAcceleratorTarget::from_device_name("Apple M4 Ultra", 2);
        assert_eq!(t.tier, DeviceTier::Ultra);
        assert_eq!(t.die_count, 2);
        assert_eq!(t.ane_cores, 32);
    }
}
