//! Backend-neutral ownership for a paged KV cache.
//!
//! A physical [`BlockId`] denotes a *bundle* containing the corresponding K/V
//! page in every model layer. The allocator owns only page-table metadata; a
//! Metal, CUDA, or other backend owns the bytes and uses [`CowPageCopy`] to
//! copy a shared partial tail before it is modified.
//!
//! Chain handles are generational. Reusing a request id or a chain slot cannot
//! make a stale completion release pages belonging to a newer request.

use std::collections::HashMap;
use std::fmt;

use rvllm_core::{BlockId, ReqId};

/// Cache ABI used by the first production Apple page-table layout.
pub const APPLE_KV_ABI_V1: u16 = 1;

/// Apple v1 has exactly 32 token positions in every logical KV page.
pub const APPLE_KV_PAGE_SIZE: u32 = 32;

/// Versioned pool configuration. Apple v1 deliberately rejects any page size
/// other than 32 so cache records and kernels cannot silently disagree.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PagedKvConfig {
    pub abi_version: u16,
    pub page_size: u32,
    pub total_pages: u32,
    pub max_chains: u32,
}

impl PagedKvConfig {
    pub const fn apple_v1(total_pages: u32, max_chains: u32) -> Self {
        Self {
            abi_version: APPLE_KV_ABI_V1,
            page_size: APPLE_KV_PAGE_SIZE,
            total_pages,
            max_chains,
        }
    }

    pub fn validate(self) -> Result<Self, PagedKvError> {
        if self.abi_version != APPLE_KV_ABI_V1 {
            return Err(PagedKvError::UnsupportedAbi {
                got: self.abi_version,
                supported: APPLE_KV_ABI_V1,
            });
        }
        if self.page_size != APPLE_KV_PAGE_SIZE {
            return Err(PagedKvError::InvalidPageSize {
                abi_version: self.abi_version,
                got: self.page_size,
                required: APPLE_KV_PAGE_SIZE,
            });
        }
        if self.total_pages == 0 {
            return Err(PagedKvError::InvalidCapacity {
                field: "total_pages",
                got: 0,
            });
        }
        if self.max_chains == 0 {
            return Err(PagedKvError::InvalidCapacity {
                field: "max_chains",
                got: 0,
            });
        }
        Ok(self)
    }
}

/// Stable identity of a chain-table slot.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[repr(transparent)]
pub struct KvChainId(pub u32);

/// A chain identity plus the generation that currently occupies its slot.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct KvChainHandle {
    pub id: KvChainId,
    pub generation: u64,
}

/// Opaque, generational-by-uniqueness lease retaining one physical page.
///
/// Page leases are used by immutable T1 prompt-cache entries. They are not
/// request ownership: attaching a cached prefix creates a request-owned chain
/// and increments the page references independently of these leases.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[repr(transparent)]
pub struct KvPageLease(pub u64);

/// Read-only page-table view valid for the borrow of the pool.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KvChainView<'a> {
    pub handle: KvChainHandle,
    pub owner: ReqId,
    pub token_len: u32,
    pub pages: &'a [BlockId],
}

/// Device-copy work required to make a shared, partially-filled tail private.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CowPageCopy {
    pub source: BlockId,
    pub destination: BlockId,
    pub valid_tokens: u32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CowTailResult {
    /// The chain is empty, ends at a page boundary, or already owns its tail.
    AlreadyWritable,
    /// The copy callback succeeded and the chain now references `destination`.
    Copied {
        source: BlockId,
        destination: BlockId,
        valid_tokens: u32,
    },
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PagedKvStats {
    pub total_pages: u32,
    pub used_pages: u32,
    pub free_pages: u32,
    pub active_chains: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PagedKvError {
    UnsupportedAbi {
        got: u16,
        supported: u16,
    },
    InvalidPageSize {
        abi_version: u16,
        got: u32,
        required: u32,
    },
    InvalidCapacity {
        field: &'static str,
        got: u32,
    },
    PagePoolExhausted {
        needed_pages: u32,
        free_pages: u32,
    },
    ChainPoolExhausted {
        max_chains: u32,
    },
    RequestAlreadyOwnsChain {
        owner: ReqId,
        chain: KvChainHandle,
    },
    RequestDoesNotOwnChain {
        owner: ReqId,
    },
    StaleChain {
        handle: KvChainHandle,
    },
    PrefixOutOfRange {
        requested_tokens: u32,
        available_tokens: u32,
    },
    TokenLengthOverflow {
        current_tokens: u32,
        additional_tokens: u32,
    },
    SharedPartialTail {
        page: BlockId,
        ref_count: u32,
    },
    RefCountOverflow {
        page: BlockId,
    },
    UnalignedPrefix {
        tokens: u32,
        page_size: u32,
    },
    UnknownPageLease {
        lease: KvPageLease,
    },
    DuplicatePageLease {
        lease: KvPageLease,
    },
    InvariantViolation {
        reason: &'static str,
    },
}

impl fmt::Display for PagedKvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedAbi { got, supported } => {
                write!(
                    f,
                    "unsupported paged-KV ABI {got}; supported ABI is {supported}"
                )
            }
            Self::InvalidPageSize {
                abi_version,
                got,
                required,
            } => write!(
                f,
                "paged-KV ABI {abi_version} requires page size {required}, got {got}"
            ),
            Self::InvalidCapacity { field, got } => {
                write!(f, "paged-KV capacity {field} must be non-zero, got {got}")
            }
            Self::PagePoolExhausted {
                needed_pages,
                free_pages,
            } => write!(
                f,
                "paged-KV pool exhausted: need {needed_pages} pages, have {free_pages}"
            ),
            Self::ChainPoolExhausted { max_chains } => {
                write!(f, "paged-KV chain pool exhausted at {max_chains} chains")
            }
            Self::RequestAlreadyOwnsChain { owner, chain } => write!(
                f,
                "request {owner} already owns KV chain {:?} generation {}",
                chain.id, chain.generation
            ),
            Self::RequestDoesNotOwnChain { owner } => {
                write!(f, "request {owner} does not own a KV chain")
            }
            Self::StaleChain { handle } => write!(
                f,
                "stale KV chain {:?} generation {}",
                handle.id, handle.generation
            ),
            Self::PrefixOutOfRange {
                requested_tokens,
                available_tokens,
            } => write!(
                f,
                "KV prefix length {requested_tokens} exceeds source length {available_tokens}"
            ),
            Self::TokenLengthOverflow {
                current_tokens,
                additional_tokens,
            } => write!(
                f,
                "KV token length overflow: {current_tokens} + {additional_tokens}"
            ),
            Self::SharedPartialTail { page, ref_count } => write!(
                f,
                "KV page {page} is a shared partial tail with {ref_count} references"
            ),
            Self::RefCountOverflow { page } => {
                write!(f, "KV page {page} reference count overflow")
            }
            Self::UnalignedPrefix { tokens, page_size } => write!(
                f,
                "KV prefix length {tokens} is not aligned to page size {page_size}"
            ),
            Self::UnknownPageLease { lease } => {
                write!(f, "unknown or released KV page lease {}", lease.0)
            }
            Self::DuplicatePageLease { lease } => {
                write!(f, "duplicate KV page lease {} in one chain", lease.0)
            }
            Self::InvariantViolation { reason } => {
                write!(f, "paged-KV allocator invariant violated: {reason}")
            }
        }
    }
}

impl std::error::Error for PagedKvError {}

#[derive(Clone, Debug)]
struct Chain {
    owner: ReqId,
    token_len: u32,
    pages: Vec<BlockId>,
}

#[derive(Clone, Debug)]
struct ChainSlot {
    generation: u64,
    chain: Option<Chain>,
}

/// Logical page allocator and request-to-chain ownership registry.
///
/// All mutating operations are single-threaded by design. The accelerator
/// worker owns the pool; callers synchronize admission at the engine edge.
#[derive(Debug)]
pub struct PagedKvPool {
    config: PagedKvConfig,
    page_refs: Vec<u32>,
    free_pages: Vec<BlockId>,
    chain_slots: Vec<ChainSlot>,
    free_chain_slots: Vec<KvChainId>,
    owner_chains: HashMap<ReqId, KvChainHandle>,
    page_leases: HashMap<KvPageLease, BlockId>,
    next_page_lease: u64,
}

impl PagedKvPool {
    pub fn new(config: PagedKvConfig) -> Result<Self, PagedKvError> {
        let config = config.validate()?;
        let free_pages = (0..config.total_pages).rev().map(BlockId).collect();
        let chain_slots = (0..config.max_chains)
            .map(|_| ChainSlot {
                generation: 1,
                chain: None,
            })
            .collect();
        let free_chain_slots = (0..config.max_chains).rev().map(KvChainId).collect();
        Ok(Self {
            config,
            page_refs: vec![0; config.total_pages as usize],
            free_pages,
            chain_slots,
            free_chain_slots,
            owner_chains: HashMap::new(),
            page_leases: HashMap::new(),
            next_page_lease: 1,
        })
    }

    pub fn config(&self) -> PagedKvConfig {
        self.config
    }

    pub fn stats(&self) -> PagedKvStats {
        let free_pages = self.free_pages.len() as u32;
        PagedKvStats {
            total_pages: self.config.total_pages,
            used_pages: self.config.total_pages - free_pages,
            free_pages,
            active_chains: self.owner_chains.len() as u32,
        }
    }

    pub fn chain_for_owner(&self, owner: ReqId) -> Option<KvChainHandle> {
        self.owner_chains.get(&owner).copied()
    }

    pub fn view(&self, handle: KvChainHandle) -> Result<KvChainView<'_>, PagedKvError> {
        let chain = self.chain(handle)?;
        Ok(KvChainView {
            handle,
            owner: chain.owner,
            token_len: chain.token_len,
            pages: &chain.pages,
        })
    }

    pub fn page_ref_count(&self, page: BlockId) -> Option<u32> {
        self.page_refs.get(page.0 as usize).copied()
    }

    /// Retains every complete page in `prefix_tokens` for an immutable cache
    /// entry. The operation is atomic: no references change if validation or
    /// lease-id allocation fails.
    pub fn retain_complete_prefix(
        &mut self,
        handle: KvChainHandle,
        prefix_tokens: u32,
    ) -> Result<Vec<KvPageLease>, PagedKvError> {
        if prefix_tokens % self.config.page_size != 0 {
            return Err(PagedKvError::UnalignedPrefix {
                tokens: prefix_tokens,
                page_size: self.config.page_size,
            });
        }
        let chain = self.chain(handle)?;
        if prefix_tokens > chain.token_len {
            return Err(PagedKvError::PrefixOutOfRange {
                requested_tokens: prefix_tokens,
                available_tokens: chain.token_len,
            });
        }
        let pages = chain.pages[..(prefix_tokens / self.config.page_size) as usize].to_vec();
        for page in &pages {
            if self.page_refs[page.0 as usize] == u32::MAX {
                return Err(PagedKvError::RefCountOverflow { page: *page });
            }
        }
        let leases = self.reserve_lease_ids(pages.len())?;
        for (&lease, &page) in leases.iter().zip(&pages) {
            self.page_refs[page.0 as usize] += 1;
            self.page_leases.insert(lease, page);
        }
        Ok(leases)
    }

    /// Creates a request-owned chain that shares immutable, page-aligned T1
    /// pages. The cache leases remain valid and independently releasable.
    pub fn allocate_chain_from_leases(
        &mut self,
        owner: ReqId,
        leases: &[KvPageLease],
        token_len: u32,
    ) -> Result<KvChainHandle, PagedKvError> {
        self.ensure_owner_available(owner)?;
        if token_len % self.config.page_size != 0 {
            return Err(PagedKvError::UnalignedPrefix {
                tokens: token_len,
                page_size: self.config.page_size,
            });
        }
        if leases.len() != pages_for_tokens(token_len, self.config.page_size) as usize {
            return Err(PagedKvError::InvariantViolation {
                reason: "page leases do not cover the requested token length",
            });
        }
        let mut seen = std::collections::HashSet::with_capacity(leases.len());
        let mut pages = Vec::with_capacity(leases.len());
        for &lease in leases {
            if !seen.insert(lease) {
                return Err(PagedKvError::DuplicatePageLease { lease });
            }
            let page = self
                .page_leases
                .get(&lease)
                .copied()
                .ok_or(PagedKvError::UnknownPageLease { lease })?;
            if self.page_refs[page.0 as usize] == u32::MAX {
                return Err(PagedKvError::RefCountOverflow { page });
            }
            pages.push(page);
        }
        let chain_id =
            self.free_chain_slots
                .last()
                .copied()
                .ok_or(PagedKvError::ChainPoolExhausted {
                    max_chains: self.config.max_chains,
                })?;

        for page in &pages {
            self.page_refs[page.0 as usize] += 1;
        }
        self.free_chain_slots.pop();
        let slot = &mut self.chain_slots[chain_id.0 as usize];
        let handle = KvChainHandle {
            id: chain_id,
            generation: slot.generation,
        };
        slot.chain = Some(Chain {
            owner,
            token_len,
            pages,
        });
        self.owner_chains.insert(owner, handle);
        Ok(handle)
    }

    /// Releases the cache's reference to a physical page. Request chains that
    /// already share it remain valid.
    pub fn release_page_lease(&mut self, lease: KvPageLease) -> Result<(), PagedKvError> {
        let page = self
            .page_leases
            .remove(&lease)
            .ok_or(PagedKvError::UnknownPageLease { lease })?;
        self.release_page(page)
    }

    pub fn page_for_lease(&self, lease: KvPageLease) -> Result<BlockId, PagedKvError> {
        self.page_leases
            .get(&lease)
            .copied()
            .ok_or(PagedKvError::UnknownPageLease { lease })
    }

    /// Creates a request-owned chain and reserves all initial pages atomically.
    pub fn allocate_chain(
        &mut self,
        owner: ReqId,
        initial_tokens: u32,
    ) -> Result<KvChainHandle, PagedKvError> {
        self.ensure_owner_available(owner)?;
        let page_count = pages_for_tokens(initial_tokens, self.config.page_size);
        self.ensure_capacity(page_count)?;
        let chain_id =
            self.free_chain_slots
                .last()
                .copied()
                .ok_or(PagedKvError::ChainPoolExhausted {
                    max_chains: self.config.max_chains,
                })?;

        // All fallible validation is complete. Mutations below form the commit.
        let pages = self.take_free_pages(page_count);
        let popped = self.free_chain_slots.pop();
        debug_assert_eq!(popped, Some(chain_id));
        let slot = &mut self.chain_slots[chain_id.0 as usize];
        let handle = KvChainHandle {
            id: chain_id,
            generation: slot.generation,
        };
        slot.chain = Some(Chain {
            owner,
            token_len: initial_tokens,
            pages,
        });
        self.owner_chains.insert(owner, handle);
        Ok(handle)
    }

    /// Shares a source prefix with a new request. An arbitrary partial prefix
    /// is allowed; callers must invoke [`Self::make_tail_writable`] before
    /// appending to a shared partial page.
    pub fn fork_prefix(
        &mut self,
        source: KvChainHandle,
        new_owner: ReqId,
        prefix_tokens: u32,
    ) -> Result<KvChainHandle, PagedKvError> {
        self.ensure_owner_available(new_owner)?;
        let source_chain = self.chain(source)?;
        if prefix_tokens > source_chain.token_len {
            return Err(PagedKvError::PrefixOutOfRange {
                requested_tokens: prefix_tokens,
                available_tokens: source_chain.token_len,
            });
        }
        let page_count = pages_for_tokens(prefix_tokens, self.config.page_size) as usize;
        let pages = source_chain.pages[..page_count].to_vec();
        for page in &pages {
            let refs = self.page_refs[page.0 as usize];
            if refs == u32::MAX {
                return Err(PagedKvError::RefCountOverflow { page: *page });
            }
        }
        let chain_id =
            self.free_chain_slots
                .last()
                .copied()
                .ok_or(PagedKvError::ChainPoolExhausted {
                    max_chains: self.config.max_chains,
                })?;

        // Commit only after every reference and chain-slot check succeeds.
        for page in &pages {
            self.page_refs[page.0 as usize] += 1;
        }
        let popped = self.free_chain_slots.pop();
        debug_assert_eq!(popped, Some(chain_id));
        let slot = &mut self.chain_slots[chain_id.0 as usize];
        let handle = KvChainHandle {
            id: chain_id,
            generation: slot.generation,
        };
        slot.chain = Some(Chain {
            owner: new_owner,
            token_len: prefix_tokens,
            pages,
        });
        self.owner_chains.insert(new_owner, handle);
        Ok(handle)
    }

    pub fn fork_chain(
        &mut self,
        source: KvChainHandle,
        new_owner: ReqId,
    ) -> Result<KvChainHandle, PagedKvError> {
        let token_len = self.chain(source)?.token_len;
        self.fork_prefix(source, new_owner, token_len)
    }

    /// Extends logical length and reserves any additional physical pages as a
    /// transaction. On exhaustion the chain and pool remain unchanged.
    pub fn append_tokens(
        &mut self,
        handle: KvChainHandle,
        additional_tokens: u32,
    ) -> Result<(), PagedKvError> {
        if additional_tokens == 0 {
            self.chain(handle)?;
            return Ok(());
        }
        let chain = self.chain(handle)?;
        let new_len = chain.token_len.checked_add(additional_tokens).ok_or(
            PagedKvError::TokenLengthOverflow {
                current_tokens: chain.token_len,
                additional_tokens,
            },
        )?;
        if chain.token_len % self.config.page_size != 0 {
            let tail = chain
                .pages
                .last()
                .copied()
                .ok_or(PagedKvError::InvariantViolation {
                    reason: "non-empty partial chain has no tail page",
                })?;
            let refs = self.page_refs[tail.0 as usize];
            if refs > 1 {
                return Err(PagedKvError::SharedPartialTail {
                    page: tail,
                    ref_count: refs,
                });
            }
        }
        let current_pages = chain.pages.len() as u32;
        let needed_total = pages_for_tokens(new_len, self.config.page_size);
        let additional_pages = needed_total - current_pages;
        self.ensure_capacity(additional_pages)?;

        let pages = self.take_free_pages(additional_pages);
        let chain = self.chain_mut(handle)?;
        chain.pages.extend(pages);
        chain.token_len = new_len;
        Ok(())
    }

    /// Roll a chain back to an earlier logical length, releasing every page
    /// that is no longer covered by the prefix. This is primarily the abort
    /// half of an engine launch transaction: pages are reserved before the
    /// accelerator sees a handoff, then returned if launch or collection
    /// fails. The retained prefix (including a partial tail) is unchanged.
    pub fn truncate_tokens(
        &mut self,
        handle: KvChainHandle,
        new_token_len: u32,
    ) -> Result<(), PagedKvError> {
        let chain = self.chain(handle)?;
        if new_token_len > chain.token_len {
            return Err(PagedKvError::PrefixOutOfRange {
                requested_tokens: new_token_len,
                available_tokens: chain.token_len,
            });
        }
        if new_token_len == chain.token_len {
            return Ok(());
        }

        let retained_pages = pages_for_tokens(new_token_len, self.config.page_size) as usize;
        let released = chain.pages[retained_pages..].to_vec();
        for page in &released {
            let Some(ref_count) = self.page_refs.get(page.0 as usize) else {
                return Err(PagedKvError::InvariantViolation {
                    reason: "chain references an out-of-range page",
                });
            };
            if *ref_count == 0 {
                return Err(PagedKvError::InvariantViolation {
                    reason: "chain page has zero reference count",
                });
            }
        }

        let chain = self.chain_mut(handle)?;
        chain.pages.truncate(retained_pages);
        chain.token_len = new_token_len;
        for page in released {
            self.release_page(page)?;
        }
        Ok(())
    }

    /// Makes a shared partial tail private and invokes `copy_page` for the
    /// backend-owned bytes. Allocation and ownership changes are committed
    /// only if the copy succeeds; a copy error releases the reserved page.
    pub fn make_tail_writable<E, F>(
        &mut self,
        handle: KvChainHandle,
        copy_page: F,
    ) -> Result<CowTailResult, CowTailError<E>>
    where
        F: FnOnce(CowPageCopy) -> Result<(), E>,
    {
        let chain = self.chain(handle).map_err(CowTailError::Allocator)?;
        let valid_tokens = chain.token_len % self.config.page_size;
        if valid_tokens == 0 {
            return Ok(CowTailResult::AlreadyWritable);
        }
        let source = chain.pages.last().copied().ok_or_else(|| {
            CowTailError::Allocator(PagedKvError::InvariantViolation {
                reason: "partial chain has no tail page",
            })
        })?;
        if self.page_refs[source.0 as usize] <= 1 {
            return Ok(CowTailResult::AlreadyWritable);
        }
        self.ensure_capacity(1).map_err(CowTailError::Allocator)?;
        let destination = self.take_free_pages(1)[0];
        let copy = CowPageCopy {
            source,
            destination,
            valid_tokens,
        };
        if let Err(error) = copy_page(copy) {
            self.release_page(destination)
                .map_err(CowTailError::Allocator)?;
            return Err(CowTailError::Copy(error));
        }

        // The callback cannot borrow this pool, so the validated chain and
        // source tail are unchanged. Commit the new page table, then release
        // the old reference.
        let chain = self.chain_mut(handle).map_err(CowTailError::Allocator)?;
        let tail = chain.pages.last_mut().ok_or_else(|| {
            CowTailError::Allocator(PagedKvError::InvariantViolation {
                reason: "partial chain lost its tail during COW",
            })
        })?;
        if *tail != source {
            self.release_page(destination)
                .map_err(CowTailError::Allocator)?;
            return Err(CowTailError::Allocator(PagedKvError::InvariantViolation {
                reason: "partial chain tail changed during COW",
            }));
        }
        *tail = destination;
        self.release_page(source).map_err(CowTailError::Allocator)?;
        Ok(CowTailResult::Copied {
            source,
            destination,
            valid_tokens,
        })
    }

    /// Releases a chain exactly once. A stale second release is rejected and
    /// cannot alter refcounts of a newer occupant of the same chain slot.
    pub fn release_chain(&mut self, handle: KvChainHandle) -> Result<(), PagedKvError> {
        let chain = self.chain(handle)?;
        if self.owner_chains.get(&chain.owner).copied() != Some(handle) {
            return Err(PagedKvError::InvariantViolation {
                reason: "request ownership map disagrees with chain slot",
            });
        }
        let mut releases: HashMap<BlockId, u32> = HashMap::new();
        for page in &chain.pages {
            *releases.entry(*page).or_insert(0) += 1;
        }
        for (page, count) in &releases {
            let Some(ref_count) = self.page_refs.get(page.0 as usize) else {
                return Err(PagedKvError::InvariantViolation {
                    reason: "chain references an out-of-range page",
                });
            };
            if ref_count < count {
                return Err(PagedKvError::InvariantViolation {
                    reason: "chain page reference count is too small",
                });
            }
        }

        // Commit after all ownership and reference validation succeeds.
        let owner = chain.owner;
        let slot = &mut self.chain_slots[handle.id.0 as usize];
        let removed = slot.chain.take();
        debug_assert!(removed.is_some());
        self.owner_chains.remove(&owner);
        for (page, count) in releases {
            let refs = &mut self.page_refs[page.0 as usize];
            *refs -= count;
            if *refs == 0 {
                self.free_pages.push(page);
            }
        }
        slot.generation = next_generation(slot.generation);
        self.free_chain_slots.push(handle.id);
        Ok(())
    }

    /// Cancellation by request id is idempotent. `Ok(false)` means the
    /// request had already completed/cancelled and no allocator state changed.
    pub fn cancel_request(&mut self, owner: ReqId) -> Result<bool, PagedKvError> {
        let Some(handle) = self.owner_chains.get(&owner).copied() else {
            return Ok(false);
        };
        self.release_chain(handle)?;
        Ok(true)
    }

    fn ensure_owner_available(&self, owner: ReqId) -> Result<(), PagedKvError> {
        if let Some(chain) = self.owner_chains.get(&owner).copied() {
            return Err(PagedKvError::RequestAlreadyOwnsChain { owner, chain });
        }
        Ok(())
    }

    fn ensure_capacity(&self, needed_pages: u32) -> Result<(), PagedKvError> {
        let free_pages = self.free_pages.len() as u32;
        if needed_pages > free_pages {
            return Err(PagedKvError::PagePoolExhausted {
                needed_pages,
                free_pages,
            });
        }
        Ok(())
    }

    fn chain(&self, handle: KvChainHandle) -> Result<&Chain, PagedKvError> {
        let Some(slot) = self.chain_slots.get(handle.id.0 as usize) else {
            return Err(PagedKvError::StaleChain { handle });
        };
        if slot.generation != handle.generation {
            return Err(PagedKvError::StaleChain { handle });
        }
        slot.chain
            .as_ref()
            .ok_or(PagedKvError::StaleChain { handle })
    }

    fn chain_mut(&mut self, handle: KvChainHandle) -> Result<&mut Chain, PagedKvError> {
        let Some(slot) = self.chain_slots.get_mut(handle.id.0 as usize) else {
            return Err(PagedKvError::StaleChain { handle });
        };
        if slot.generation != handle.generation {
            return Err(PagedKvError::StaleChain { handle });
        }
        slot.chain
            .as_mut()
            .ok_or(PagedKvError::StaleChain { handle })
    }

    fn take_free_pages(&mut self, count: u32) -> Vec<BlockId> {
        debug_assert!(count as usize <= self.free_pages.len());
        let mut pages = Vec::with_capacity(count as usize);
        for _ in 0..count {
            if let Some(page) = self.free_pages.pop() {
                debug_assert_eq!(self.page_refs[page.0 as usize], 0);
                self.page_refs[page.0 as usize] = 1;
                pages.push(page);
            }
        }
        pages
    }

    fn reserve_lease_ids(&mut self, count: usize) -> Result<Vec<KvPageLease>, PagedKvError> {
        let mut ids = Vec::with_capacity(count);
        let mut next = self.next_page_lease;
        for _ in 0..count {
            let start = next;
            loop {
                if next != 0 && !self.page_leases.contains_key(&KvPageLease(next)) {
                    ids.push(KvPageLease(next));
                    next = next.wrapping_add(1);
                    break;
                }
                next = next.wrapping_add(1);
                if next == start {
                    return Err(PagedKvError::InvariantViolation {
                        reason: "KV page lease id space exhausted",
                    });
                }
            }
        }
        self.next_page_lease = next;
        Ok(ids)
    }

    fn release_page(&mut self, page: BlockId) -> Result<(), PagedKvError> {
        let Some(refs) = self.page_refs.get_mut(page.0 as usize) else {
            return Err(PagedKvError::InvariantViolation {
                reason: "chain references an out-of-range page",
            });
        };
        if *refs == 0 {
            return Err(PagedKvError::InvariantViolation {
                reason: "attempted to release a free page",
            });
        }
        *refs -= 1;
        if *refs == 0 {
            self.free_pages.push(page);
        }
        Ok(())
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum CowTailError<E> {
    Allocator(PagedKvError),
    Copy(E),
}

impl<E: fmt::Display> fmt::Display for CowTailError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allocator(error) => error.fmt(f),
            Self::Copy(error) => write!(f, "KV tail copy failed: {error}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for CowTailError<E> {}

fn pages_for_tokens(tokens: u32, page_size: u32) -> u32 {
    tokens / page_size + u32::from(tokens % page_size != 0)
}

fn next_generation(current: u64) -> u64 {
    let next = current.wrapping_add(1);
    if next == 0 {
        1
    } else {
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(total_pages: u32, max_chains: u32) -> PagedKvPool {
        PagedKvPool::new(PagedKvConfig::apple_v1(total_pages, max_chains))
            .unwrap_or_else(|error| panic!("test pool must be valid: {error}"))
    }

    #[test]
    fn apple_v1_rejects_non_32_page_size() {
        let error = PagedKvPool::new(PagedKvConfig {
            abi_version: APPLE_KV_ABI_V1,
            page_size: 16,
            total_pages: 1,
            max_chains: 1,
        })
        .unwrap_err();
        assert_eq!(
            error,
            PagedKvError::InvalidPageSize {
                abi_version: APPLE_KV_ABI_V1,
                got: 16,
                required: 32,
            }
        );
    }

    #[test]
    fn allocation_is_transactional_on_page_exhaustion() {
        let mut pool = pool(2, 3);
        let before = pool.stats();
        let error = pool.allocate_chain(ReqId(7), 65).unwrap_err();
        assert_eq!(
            error,
            PagedKvError::PagePoolExhausted {
                needed_pages: 3,
                free_pages: 2,
            }
        );
        assert_eq!(pool.stats(), before);
        assert_eq!(pool.chain_for_owner(ReqId(7)), None);

        let chain = pool
            .allocate_chain(ReqId(8), 64)
            .unwrap_or_else(|error| panic!("all pages should still be free: {error}"));
        assert_eq!(pool.view(chain).map(|view| view.pages.len()), Ok(2));
    }

    #[test]
    fn append_rolls_back_when_all_pages_are_not_available() {
        let mut pool = pool(3, 2);
        let chain = pool
            .allocate_chain(ReqId(1), 32)
            .unwrap_or_else(|error| panic!("allocation failed: {error}"));
        let before = pool
            .view(chain)
            .map(|view| (view.token_len, view.pages.to_vec()));
        let stats_before = pool.stats();
        assert!(matches!(
            pool.append_tokens(chain, 65),
            Err(PagedKvError::PagePoolExhausted {
                needed_pages: 3,
                free_pages: 2
            })
        ));
        assert_eq!(
            pool.view(chain)
                .map(|view| (view.token_len, view.pages.to_vec())),
            before
        );
        assert_eq!(pool.stats(), stats_before);
    }

    #[test]
    fn truncate_restores_length_and_releases_only_suffix_pages() {
        let mut pool = pool(5, 2);
        let chain = pool.allocate_chain(ReqId(1), 17).unwrap();
        let first_page = pool.view(chain).unwrap().pages[0];
        pool.append_tokens(chain, 64).unwrap();
        assert_eq!(pool.view(chain).unwrap().token_len, 81);
        assert_eq!(pool.stats().used_pages, 3);

        pool.truncate_tokens(chain, 17).unwrap();
        let view = pool.view(chain).unwrap();
        assert_eq!(view.token_len, 17);
        assert_eq!(view.pages, &[first_page]);
        assert_eq!(pool.stats().used_pages, 1);
    }

    #[test]
    fn request_ownership_does_not_depend_on_batch_order() {
        let mut pool = pool(8, 4);
        let a = pool
            .allocate_chain(ReqId(10), 33)
            .unwrap_or_else(|error| panic!("allocation failed: {error}"));
        let b = pool
            .allocate_chain(ReqId(20), 1)
            .unwrap_or_else(|error| panic!("allocation failed: {error}"));
        let a_pages = pool.view(a).map(|view| view.pages.to_vec());
        let b_pages = pool.view(b).map(|view| view.pages.to_vec());

        // Simulate two scheduler batches with opposite sequence order.
        let first_batch = [ReqId(10), ReqId(20)];
        let second_batch = [ReqId(20), ReqId(10)];
        for owner in first_batch.into_iter().chain(second_batch) {
            let handle = pool
                .chain_for_owner(owner)
                .unwrap_or_else(|| panic!("request {owner} lost its chain"));
            if owner == ReqId(10) {
                assert_eq!(pool.view(handle).map(|view| view.pages.to_vec()), a_pages);
            } else {
                assert_eq!(pool.view(handle).map(|view| view.pages.to_vec()), b_pages);
            }
        }
    }

    #[test]
    fn fork_shares_pages_and_cow_makes_partial_tail_private() {
        let mut pool = pool(6, 3);
        let source = pool
            .allocate_chain(ReqId(1), 33)
            .unwrap_or_else(|error| panic!("allocation failed: {error}"));
        let fork = pool
            .fork_chain(source, ReqId(2))
            .unwrap_or_else(|error| panic!("fork failed: {error}"));
        let source_pages = pool
            .view(source)
            .map(|view| view.pages.to_vec())
            .unwrap_or_else(|error| panic!("view failed: {error}"));
        assert_eq!(pool.page_ref_count(source_pages[0]), Some(2));
        assert_eq!(pool.page_ref_count(source_pages[1]), Some(2));
        assert!(matches!(
            pool.append_tokens(fork, 1),
            Err(PagedKvError::SharedPartialTail { .. })
        ));

        let mut observed = None;
        let cow = pool
            .make_tail_writable(fork, |copy| {
                observed = Some(copy);
                Ok::<(), &'static str>(())
            })
            .unwrap_or_else(|error| panic!("COW failed: {error}"));
        let copy = observed.unwrap_or_else(|| panic!("copy callback was not invoked"));
        assert_eq!(copy.source, source_pages[1]);
        assert_eq!(copy.valid_tokens, 1);
        assert!(matches!(cow, CowTailResult::Copied { .. }));

        let fork_pages = pool
            .view(fork)
            .map(|view| view.pages.to_vec())
            .unwrap_or_else(|error| panic!("view failed: {error}"));
        assert_eq!(fork_pages[0], source_pages[0]);
        assert_ne!(fork_pages[1], source_pages[1]);
        assert_eq!(pool.page_ref_count(source_pages[1]), Some(1));
        assert_eq!(pool.page_ref_count(copy.destination), Some(1));
        assert_eq!(pool.append_tokens(fork, 1), Ok(()));
    }

    #[test]
    fn cow_copy_failure_rolls_back_reserved_page() {
        let mut pool = pool(3, 2);
        let source = pool
            .allocate_chain(ReqId(1), 17)
            .unwrap_or_else(|error| panic!("allocation failed: {error}"));
        let fork = pool
            .fork_chain(source, ReqId(2))
            .unwrap_or_else(|error| panic!("fork failed: {error}"));
        let before_pages = pool.view(fork).map(|view| view.pages.to_vec());
        let before_stats = pool.stats();
        let result = pool.make_tail_writable(fork, |_copy| Err("device copy failed"));
        assert_eq!(result, Err(CowTailError::Copy("device copy failed")));
        assert_eq!(
            pool.view(fork).map(|view| view.pages.to_vec()),
            before_pages
        );
        assert_eq!(pool.stats(), before_stats);
    }

    #[test]
    fn cancel_is_idempotent_and_stale_release_cannot_free_new_chain() {
        let mut pool = pool(2, 1);
        let old = pool
            .allocate_chain(ReqId(4), 1)
            .unwrap_or_else(|error| panic!("allocation failed: {error}"));
        assert_eq!(pool.cancel_request(ReqId(4)), Ok(true));
        assert_eq!(pool.cancel_request(ReqId(4)), Ok(false));

        let new = pool
            .allocate_chain(ReqId(4), 1)
            .unwrap_or_else(|error| panic!("re-allocation failed: {error}"));
        assert_eq!(old.id, new.id);
        assert_ne!(old.generation, new.generation);
        assert_eq!(
            pool.release_chain(old),
            Err(PagedKvError::StaleChain { handle: old })
        );
        assert_eq!(pool.chain_for_owner(ReqId(4)), Some(new));
        assert_eq!(pool.stats().used_pages, 1);
    }

    #[test]
    fn full_page_prefix_can_append_without_cow() {
        let mut pool = pool(4, 2);
        let source = pool
            .allocate_chain(ReqId(1), 64)
            .unwrap_or_else(|error| panic!("allocation failed: {error}"));
        let fork = pool
            .fork_prefix(source, ReqId(2), 32)
            .unwrap_or_else(|error| panic!("fork failed: {error}"));
        assert_eq!(pool.append_tokens(fork, 1), Ok(()));
        let view = pool
            .view(fork)
            .unwrap_or_else(|error| panic!("view failed: {error}"));
        assert_eq!(view.token_len, 33);
        assert_eq!(view.pages.len(), 2);
    }
}
