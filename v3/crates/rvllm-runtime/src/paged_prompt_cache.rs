//! Safe bridge between exact prompt-cache metadata and request-owned paged KV.
//!
//! T1 entries retain real physical pages through [`KvPageLease`]. A cache hit
//! adds independent request ownership, so eviction and cancellation can race
//! without dangling page-table entries. T2/T3 bytes are restored only through
//! an installed backend implementation; the default implementation fails
//! closed and never fabricates KV bytes.

use std::fmt;
use std::sync::Arc;

use rvllm_core::{BlockId, ReqId, TokenId};

use crate::paged_kv::{
    CowPageCopy, CowTailError, CowTailResult, KvChainHandle, KvPageLease, PagedKvError,
    PagedKvPool, APPLE_KV_PAGE_SIZE,
};
use crate::persistent_prompt_cache::{
    PersistentCacheError, PersistentCacheRecord, PersistentPromptCache, PersistentStoreOutcome,
};
use crate::prompt_cache::{
    reusable_prompt_tokens, Admission, CacheError, CacheIdentity, CacheLookup, CacheNamespace,
    CachedPage, HotPageHandle, MemoryPressure, PageResidency, PressureRelease, PromptCache,
    PromptCacheConfig, WarmPage,
};

/// Device-owned page-byte operations. Implementations must copy every model
/// layer's K and V bytes for one physical page and must not return before the
/// copy is safe for the allocator operation that requested it.
pub trait KvPageIo {
    /// Exact serialized byte length of one physical page bundle.
    fn page_bytes(&self) -> Option<usize>;

    fn capture_page(&mut self, page: BlockId) -> Result<Arc<[u8]>, KvPageIoError>;
    fn restore_page(&mut self, page: BlockId, bytes: &[u8]) -> Result<(), KvPageIoError>;
    fn copy_page(&mut self, copy: CowPageCopy) -> Result<(), KvPageIoError>;
}

/// Shipping-safe default used until a backend installs real blit/copy hooks.
#[derive(Default)]
pub struct UnavailableKvPageIo;

impl KvPageIo for UnavailableKvPageIo {
    fn page_bytes(&self) -> Option<usize> {
        None
    }

    fn capture_page(&mut self, _page: BlockId) -> Result<Arc<[u8]>, KvPageIoError> {
        Err(KvPageIoError::BackendUnavailable)
    }

    fn restore_page(&mut self, _page: BlockId, _bytes: &[u8]) -> Result<(), KvPageIoError> {
        Err(KvPageIoError::BackendUnavailable)
    }

    fn copy_page(&mut self, _copy: CowPageCopy) -> Result<(), KvPageIoError> {
        Err(KvPageIoError::BackendUnavailable)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvPageIoError {
    BackendUnavailable,
    BackendBusy,
    InvalidPageId { page: u32, total_pages: u32 },
    LayoutMismatch(&'static str),
    InvalidPageBytes { expected: usize, got: usize },
    DeviceCopyFailed(String),
}

impl fmt::Display for KvPageIoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => write!(f, "KV page-byte backend is unavailable"),
            Self::BackendBusy => write!(
                f,
                "KV page-byte backend is owned by an in-flight GPU submission"
            ),
            Self::InvalidPageId { page, total_pages } => write!(
                f,
                "KV physical page {page} is out of range for {total_pages} pages"
            ),
            Self::LayoutMismatch(reason) => write!(f, "KV physical-page layout mismatch: {reason}"),
            Self::InvalidPageBytes { expected, got } => {
                write!(f, "KV page has {got} bytes; expected exactly {expected}")
            }
            Self::DeviceCopyFailed(reason) => write!(f, "KV device copy failed: {reason}"),
        }
    }
}

impl std::error::Error for KvPageIoError {}

#[derive(Debug)]
pub enum PagedPromptCacheError {
    Metadata(CacheError),
    Allocator(PagedKvError),
    PageIo(KvPageIoError),
    Persistent(PersistentCacheError),
    Cancelled,
    MixedResidency,
    IdentityLayoutMismatch,
}

impl fmt::Display for PagedPromptCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Metadata(error) => error.fmt(f),
            Self::Allocator(error) => error.fmt(f),
            Self::PageIo(error) => error.fmt(f),
            Self::Persistent(error) => error.fmt(f),
            Self::Cancelled => write!(f, "prompt-cache operation cancelled"),
            Self::MixedResidency => write!(
                f,
                "mixed hot/warm prefix requires recomputation; no unsafe partial restore was attempted"
            ),
            Self::IdentityLayoutMismatch => write!(
                f,
                "prompt-cache KV layout identity does not match the active page pool"
            ),
        }
    }
}

impl std::error::Error for PagedPromptCacheError {}

impl From<CacheError> for PagedPromptCacheError {
    fn from(value: CacheError) -> Self {
        Self::Metadata(value)
    }
}

impl From<PagedKvError> for PagedPromptCacheError {
    fn from(value: PagedKvError) -> Self {
        Self::Allocator(value)
    }
}

impl From<KvPageIoError> for PagedPromptCacheError {
    fn from(value: KvPageIoError) -> Self {
        Self::PageIo(value)
    }
}

impl From<PersistentCacheError> for PagedPromptCacheError {
    fn from(value: PersistentCacheError) -> Self {
        Self::Persistent(value)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AttachedCacheTier {
    T1Hot,
    T2Warm,
}

/// Active T0 ownership. The request chain and metadata pin must be released
/// together with [`PagedPromptCache::release_attachment`].
#[derive(Debug, Eq, PartialEq)]
pub struct CachePrefixAttachment {
    pub chain: KvChainHandle,
    pub matched_tokens: u32,
    pub tier: AttachedCacheTier,
    metadata_lease: Option<u64>,
}

/// An exact T2 lookup whose immutable page bytes are owned independently of
/// the metadata cache. Preparing a restore performs no device I/O or page-pool
/// allocation, so callers may retain this value while draining accelerator
/// work and may safely retry or recompute after a busy backend.
#[derive(Clone, Debug)]
pub struct PreparedWarmRestore {
    matched_tokens: u32,
    pages: Vec<WarmPage>,
}

impl PreparedWarmRestore {
    #[must_use]
    pub fn matched_tokens(&self) -> u32 {
        self.matched_tokens
    }

    #[must_use]
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }
}

pub struct PagedPromptCache {
    metadata: PromptCache,
    /// Page-pool layout fingerprint. This must equal `CacheIdentity::kv_layout`
    /// for every operation, preventing cross-model physical-page attachment.
    kv_layout: [u8; 32],
    /// Immutable byte accounting bound to `kv_layout` by the runtime plan.
    page_bytes: usize,
}

impl PagedPromptCache {
    pub fn new(
        config: PromptCacheConfig,
        kv_layout: [u8; 32],
        page_bytes: usize,
    ) -> Result<Self, PagedPromptCacheError> {
        if page_bytes == 0 {
            return Err(KvPageIoError::InvalidPageBytes {
                expected: 1,
                got: 0,
            }
            .into());
        }
        Ok(Self {
            metadata: PromptCache::new(config)?,
            kv_layout,
            page_bytes,
        })
    }

    pub fn metadata(&self) -> &PromptCache {
        &self.metadata
    }

    /// Promote complete prompt pages already resident in the active pool to
    /// T1. Cache ownership is represented by allocator leases, not raw page
    /// numbers. Evicted and duplicate leases are reclaimed before return.
    pub fn promote_hot(
        &mut self,
        pool: &mut PagedKvPool,
        identity: CacheIdentity,
        prompt: &[TokenId],
        source: KvChainHandle,
    ) -> Result<Admission, PagedPromptCacheError> {
        self.validate_identity(&identity)?;
        let reusable = reusable_prompt_tokens(prompt.len());
        let reusable = u32::try_from(reusable).map_err(|_| PagedKvError::InvariantViolation {
            reason: "prompt is too long",
        })?;
        let leases = pool.retain_complete_prefix(source, reusable)?;
        let handles = leases
            .iter()
            .map(|lease| HotPageHandle {
                id: lease.0,
                bytes: self.page_bytes,
            })
            .collect();
        let admission = match self.metadata.insert_hot(identity, prompt, handles) {
            Ok(admission) => admission,
            Err(error) => {
                release_leases(pool, leases.into_iter())?;
                return Err(error.into());
            }
        };
        release_hot_handles(pool, admission.evicted_hot.iter().copied())?;
        release_hot_handles(pool, admission.unused_hot.iter().copied())?;
        Ok(admission)
    }

    /// Capture exact T2 bytes from complete prompt pages. A cancellation or
    /// device error admits nothing. The caller may also persist the returned
    /// pages through `capture_warm_and_persist`.
    pub fn capture_warm<C>(
        &mut self,
        pool: &PagedKvPool,
        io: &mut dyn KvPageIo,
        identity: CacheIdentity,
        prompt: &[TokenId],
        source: KvChainHandle,
        cancelled: C,
    ) -> Result<(Admission, Vec<WarmPage>), PagedPromptCacheError>
    where
        C: Fn() -> bool,
    {
        self.validate_identity(&identity)?;
        let expected = self.validate_io(io)?;
        let reusable = reusable_prompt_tokens(prompt.len());
        let view = pool.view(source)?;
        if reusable as u64 > u64::from(view.token_len) {
            return Err(PagedKvError::PrefixOutOfRange {
                requested_tokens: reusable as u32,
                available_tokens: view.token_len,
            }
            .into());
        }
        let mut pages = Vec::with_capacity(reusable / APPLE_KV_PAGE_SIZE as usize);
        for &page in &view.pages[..reusable / APPLE_KV_PAGE_SIZE as usize] {
            if cancelled() {
                return Err(PagedPromptCacheError::Cancelled);
            }
            let bytes = io.capture_page(page)?;
            validate_page_bytes(expected, bytes.len())?;
            pages.push(WarmPage { bytes });
        }
        if cancelled() {
            return Err(PagedPromptCacheError::Cancelled);
        }
        let admission = self.metadata.insert_warm(identity, prompt, pages.clone())?;
        Ok((admission, pages))
    }

    pub fn capture_warm_and_persist<C>(
        &mut self,
        pool: &PagedKvPool,
        io: &mut dyn KvPageIo,
        persistent: &PersistentPromptCache,
        identity: CacheIdentity,
        prompt: &[TokenId],
        source: KvChainHandle,
        cancelled: C,
    ) -> Result<(Admission, PersistentStoreOutcome), PagedPromptCacheError>
    where
        C: Fn() -> bool,
    {
        let (admission, pages) =
            self.capture_warm(pool, io, identity.clone(), prompt, source, cancelled)?;
        let stored = persistent.store_prompt(&identity, prompt, &pages)?;
        Ok((admission, stored))
    }

    /// Look up and retain the longest exact all-warm prefix without touching
    /// the page pool or device. Hot-only hits are not warm restore candidates;
    /// mixed-residency hits fail closed.
    pub fn prepare_warm_restore(
        &mut self,
        identity: &CacheIdentity,
        prompt: &[TokenId],
    ) -> Result<Option<PreparedWarmRestore>, PagedPromptCacheError> {
        self.validate_identity(identity)?;
        let hit = match self.metadata.lookup(identity, prompt) {
            CacheLookup::Miss => return Ok(None),
            CacheLookup::Hit(hit) => hit,
        };
        if hit.lowest_tier == PageResidency::Hot {
            return Ok(None);
        }
        let pages = hit
            .pages
            .into_iter()
            .map(|page| match page {
                CachedPage::Warm(page) => Ok(page),
                CachedPage::Hot(_) => Err(PagedPromptCacheError::MixedResidency),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let matched_tokens =
            u32::try_from(hit.matched_tokens).map_err(|_| PagedKvError::InvariantViolation {
                reason: "prompt is too long",
            })?;
        if matched_tokens % APPLE_KV_PAGE_SIZE != 0
            || pages.len() != matched_tokens as usize / APPLE_KV_PAGE_SIZE as usize
        {
            return Err(CacheError::PageCountMismatch.into());
        }
        for page in &pages {
            validate_page_bytes(self.page_bytes, page.bytes.len())?;
        }
        Ok(Some(PreparedWarmRestore {
            matched_tokens,
            pages,
        }))
    }

    /// Transactionally allocate and restore a previously prepared exact T2
    /// prefix. On cancellation or any page-I/O failure the new request chain
    /// is released, while `prepared` remains owned by the caller for retry or
    /// recomputation policy.
    pub fn materialize_prepared_warm<C>(
        &mut self,
        pool: &mut PagedKvPool,
        io: &mut dyn KvPageIo,
        owner: ReqId,
        prepared: &PreparedWarmRestore,
        cancelled: C,
    ) -> Result<Option<CachePrefixAttachment>, PagedPromptCacheError>
    where
        C: Fn() -> bool,
    {
        self.restore_warm_pages(
            pool,
            io,
            owner,
            prepared.matched_tokens as usize,
            &prepared.pages,
            cancelled,
        )
    }

    /// Validate and retain an authenticated T3 record as the same owned,
    /// device-independent restore plan used by T2. This performs no page-pool
    /// allocation or device I/O.
    pub fn prepare_persistent_restore(
        &self,
        identity: &CacheIdentity,
        prompt: &[TokenId],
        record: &PersistentCacheRecord,
    ) -> Result<PreparedWarmRestore, PagedPromptCacheError> {
        self.validate_identity(identity)?;
        if record.matched_tokens != reusable_prompt_tokens(prompt.len())
            || record.matched_tokens % APPLE_KV_PAGE_SIZE as usize != 0
            || record.pages.len() != record.matched_tokens / APPLE_KV_PAGE_SIZE as usize
        {
            return Err(CacheError::PageCountMismatch.into());
        }
        for page in &record.pages {
            validate_page_bytes(self.page_bytes, page.bytes.len())?;
        }
        let matched_tokens =
            u32::try_from(record.matched_tokens).map_err(|_| PagedKvError::InvariantViolation {
                reason: "prompt is too long",
            })?;
        Ok(PreparedWarmRestore {
            matched_tokens,
            pages: record.pages.clone(),
        })
    }

    /// Attach the longest exact prefix. Hot pages are shared without copying;
    /// warm pages are restored into private pages. Mixed hits fail closed so a
    /// caller can recompute instead of accidentally combining incompatible
    /// byte sources.
    pub fn attach<C>(
        &mut self,
        pool: &mut PagedKvPool,
        io: &mut dyn KvPageIo,
        owner: ReqId,
        identity: &CacheIdentity,
        prompt: &[TokenId],
        cancelled: C,
    ) -> Result<Option<CachePrefixAttachment>, PagedPromptCacheError>
    where
        C: Fn() -> bool,
    {
        self.validate_identity(identity)?;
        let hit = match self.metadata.lookup(identity, prompt) {
            CacheLookup::Miss => return Ok(None),
            CacheLookup::Hit(hit) => hit,
        };
        if cancelled() {
            return Err(PagedPromptCacheError::Cancelled);
        }
        match hit.lowest_tier {
            PageResidency::Hot => {
                let metadata_lease = self.metadata.pin(hit.match_id)?;
                let leases: Result<Vec<_>, _> = hit
                    .pages
                    .iter()
                    .map(|page| match page {
                        CachedPage::Hot(handle) => Ok(KvPageLease(handle.id)),
                        CachedPage::Warm(_) => Err(PagedPromptCacheError::MixedResidency),
                    })
                    .collect();
                let leases = match leases {
                    Ok(leases) => leases,
                    Err(error) => {
                        self.metadata.release(metadata_lease)?;
                        return Err(error);
                    }
                };
                let chain = match pool.allocate_chain_from_leases(
                    owner,
                    &leases,
                    hit.matched_tokens as u32,
                ) {
                    Ok(chain) => chain,
                    Err(error) => {
                        self.metadata.release(metadata_lease)?;
                        return Err(error.into());
                    }
                };
                Ok(Some(CachePrefixAttachment {
                    chain,
                    matched_tokens: hit.matched_tokens as u32,
                    tier: AttachedCacheTier::T1Hot,
                    metadata_lease: Some(metadata_lease),
                }))
            }
            PageResidency::Warm => {
                if hit
                    .pages
                    .iter()
                    .any(|page| matches!(page, CachedPage::Hot(_)))
                {
                    return Err(PagedPromptCacheError::MixedResidency);
                }
                let pages: Vec<_> = hit
                    .pages
                    .into_iter()
                    .map(|page| match page {
                        CachedPage::Warm(page) => Ok(page),
                        CachedPage::Hot(_) => Err(PagedPromptCacheError::MixedResidency),
                    })
                    .collect::<Result<_, _>>()?;
                self.restore_warm_pages(pool, io, owner, hit.matched_tokens, &pages, cancelled)
            }
        }
    }

    /// Restore an authenticated T3 record without first admitting it to T2.
    /// Exact identity and token-derived reusable length are revalidated here.
    pub fn restore_persistent_record<C>(
        &mut self,
        pool: &mut PagedKvPool,
        io: &mut dyn KvPageIo,
        owner: ReqId,
        identity: &CacheIdentity,
        prompt: &[TokenId],
        record: &PersistentCacheRecord,
        cancelled: C,
    ) -> Result<Option<CachePrefixAttachment>, PagedPromptCacheError>
    where
        C: Fn() -> bool,
    {
        let prepared = self.prepare_persistent_restore(identity, prompt, record)?;
        self.materialize_prepared_warm(pool, io, owner, &prepared, cancelled)
    }

    fn restore_warm_pages<C>(
        &mut self,
        pool: &mut PagedKvPool,
        io: &mut dyn KvPageIo,
        owner: ReqId,
        matched_tokens: usize,
        pages: &[WarmPage],
        cancelled: C,
    ) -> Result<Option<CachePrefixAttachment>, PagedPromptCacheError>
    where
        C: Fn() -> bool,
    {
        let expected = self.validate_io(io)?;
        if matched_tokens % APPLE_KV_PAGE_SIZE as usize != 0
            || pages.len() != matched_tokens / APPLE_KV_PAGE_SIZE as usize
        {
            return Err(CacheError::PageCountMismatch.into());
        }
        for page in pages {
            validate_page_bytes(expected, page.bytes.len())?;
        }
        let matched_tokens =
            u32::try_from(matched_tokens).map_err(|_| PagedKvError::InvariantViolation {
                reason: "prompt is too long",
            })?;
        let chain = pool.allocate_chain(owner, matched_tokens)?;
        let physical = pool.view(chain)?.pages.to_vec();
        for (&destination, page) in physical.iter().zip(pages) {
            if cancelled() {
                pool.release_chain(chain)?;
                return Err(PagedPromptCacheError::Cancelled);
            }
            if let Err(error) = io.restore_page(destination, &page.bytes) {
                pool.release_chain(chain)?;
                return Err(error.into());
            }
        }
        if cancelled() {
            pool.release_chain(chain)?;
            return Err(PagedPromptCacheError::Cancelled);
        }
        Ok(Some(CachePrefixAttachment {
            chain,
            matched_tokens,
            tier: AttachedCacheTier::T2Warm,
            metadata_lease: None,
        }))
    }

    /// Cancellation-safe T0 release. The physical request chain is dropped
    /// before its cache pin, so pages remain retained throughout reclamation.
    pub fn release_attachment(
        &mut self,
        pool: &mut PagedKvPool,
        attachment: &CachePrefixAttachment,
    ) -> Result<(), PagedPromptCacheError> {
        // Never drop the metadata pin unless the request-owned chain has been
        // reclaimed successfully. Keeping the borrowed attachment with the
        // caller makes an error observable without silently losing ownership.
        pool.release_chain(attachment.chain)?;
        if let Some(lease) = attachment.metadata_lease {
            self.metadata.release(lease)?;
        }
        Ok(())
    }

    pub fn make_tail_writable(
        &mut self,
        pool: &mut PagedKvPool,
        io: &mut dyn KvPageIo,
        chain: KvChainHandle,
    ) -> Result<CowTailResult, PagedPromptCacheError> {
        self.validate_io(io)?;
        match pool.make_tail_writable(chain, |copy| io.copy_page(copy)) {
            Ok(result) => Ok(result),
            Err(CowTailError::Allocator(error)) => Err(error.into()),
            Err(CowTailError::Copy(error)) => Err(error.into()),
        }
    }

    /// Purges unpinned T1/T2 state and immediately returns every released hot
    /// allocator lease. Active T0 attachments are preserved by metadata pins.
    pub fn handle_memory_pressure(
        &mut self,
        pool: &mut PagedKvPool,
        pressure: MemoryPressure,
    ) -> Result<PressureRelease, PagedPromptCacheError> {
        let release = self.metadata.handle_memory_pressure(pressure);
        release_hot_handles(pool, release.hot.iter().copied())?;
        Ok(release)
    }

    pub fn invalidate_identity(
        &mut self,
        pool: &mut PagedKvPool,
        identity: &CacheIdentity,
    ) -> Result<PressureRelease, PagedPromptCacheError> {
        let release = self.metadata.invalidate_identity(identity);
        release_hot_handles(pool, release.hot.iter().copied())?;
        Ok(release)
    }

    pub fn invalidate_namespace(
        &mut self,
        pool: &mut PagedKvPool,
        namespace: &CacheNamespace,
    ) -> Result<PressureRelease, PagedPromptCacheError> {
        let release = self.metadata.invalidate_namespace(namespace);
        release_hot_handles(pool, release.hot.iter().copied())?;
        Ok(release)
    }

    fn validate_identity(&self, identity: &CacheIdentity) -> Result<(), PagedPromptCacheError> {
        if identity.kv_layout != self.kv_layout {
            return Err(PagedPromptCacheError::IdentityLayoutMismatch);
        }
        Ok(())
    }

    fn validate_io(&self, io: &dyn KvPageIo) -> Result<usize, PagedPromptCacheError> {
        let got = io.page_bytes().ok_or(KvPageIoError::BackendUnavailable)?;
        validate_page_bytes(self.page_bytes, got)?;
        Ok(self.page_bytes)
    }
}

fn validate_page_bytes(expected: usize, got: usize) -> Result<(), PagedPromptCacheError> {
    if expected == 0 || expected != got {
        return Err(KvPageIoError::InvalidPageBytes { expected, got }.into());
    }
    Ok(())
}

fn release_hot_handles(
    pool: &mut PagedKvPool,
    handles: impl Iterator<Item = HotPageHandle>,
) -> Result<(), PagedPromptCacheError> {
    release_leases(pool, handles.map(|handle| KvPageLease(handle.id)))
}

fn release_leases(
    pool: &mut PagedKvPool,
    leases: impl Iterator<Item = KvPageLease>,
) -> Result<(), PagedPromptCacheError> {
    let mut first_error = None;
    for lease in leases {
        if let Err(error) = pool.release_page_lease(lease) {
            first_error.get_or_insert(error);
        }
    }
    if let Some(error) = first_error {
        return Err(error.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::paged_kv::PagedKvConfig;

    const LAYOUT: [u8; 32] = [7; 32];

    fn identity() -> CacheIdentity {
        CacheIdentity {
            namespace: CacheNamespace::new("tenant-a").unwrap(),
            model: [1; 32],
            tokenizer: [2; 32],
            adapter: None,
            kv_layout: LAYOUT,
            numeric_path: [3; 32],
            format_version: 1,
        }
    }

    fn prompt() -> Vec<TokenId> {
        (0..65).map(TokenId).collect()
    }

    fn config() -> PromptCacheConfig {
        PromptCacheConfig {
            hot_bytes: 1024,
            warm_bytes: 1024,
            protected_fraction_percent: 80,
            frequency_aging_interval: 100,
        }
    }

    #[derive(Debug)]
    struct FakeIo {
        page_bytes: Option<usize>,
        pages: Vec<Vec<u8>>,
        restores: usize,
        copies: usize,
        restore_error: Option<KvPageIoError>,
    }

    impl FakeIo {
        fn new(total_pages: usize, page_bytes: usize) -> Self {
            Self {
                page_bytes: Some(page_bytes),
                pages: (0..total_pages)
                    .map(|page| vec![page as u8 + 10; page_bytes])
                    .collect(),
                restores: 0,
                copies: 0,
                restore_error: None,
            }
        }
    }

    impl KvPageIo for FakeIo {
        fn page_bytes(&self) -> Option<usize> {
            self.page_bytes
        }

        fn capture_page(&mut self, page: BlockId) -> Result<Arc<[u8]>, KvPageIoError> {
            Ok(self.pages[page.0 as usize].clone().into())
        }

        fn restore_page(&mut self, page: BlockId, bytes: &[u8]) -> Result<(), KvPageIoError> {
            if let Some(error) = self.restore_error.clone() {
                return Err(error);
            }
            self.restores += 1;
            self.pages[page.0 as usize].copy_from_slice(bytes);
            Ok(())
        }

        fn copy_page(&mut self, copy: CowPageCopy) -> Result<(), KvPageIoError> {
            self.copies += 1;
            let bytes = self.pages[copy.source.0 as usize].clone();
            self.pages[copy.destination.0 as usize].copy_from_slice(&bytes);
            Ok(())
        }
    }

    #[test]
    fn hot_attachment_has_independent_request_and_cache_ownership() {
        let mut pool = PagedKvPool::new(PagedKvConfig::apple_v1(6, 3)).unwrap();
        let source = pool.allocate_chain(ReqId(1), 65).unwrap();
        let source_pages = pool.view(source).unwrap().pages[..2].to_vec();
        let mut cache = PagedPromptCache::new(config(), LAYOUT, 64).unwrap();
        let admitted = cache
            .promote_hot(&mut pool, identity(), &prompt(), source)
            .unwrap();
        assert_eq!(admitted.admitted_pages, 2);
        pool.release_chain(source).unwrap();
        assert_eq!(pool.stats().used_pages, 2);

        let mut unavailable = UnavailableKvPageIo;
        let attached = cache
            .attach(
                &mut pool,
                &mut unavailable,
                ReqId(2),
                &identity(),
                &prompt(),
                || false,
            )
            .unwrap()
            .unwrap();
        assert_eq!(attached.tier, AttachedCacheTier::T1Hot);
        assert_eq!(pool.view(attached.chain).unwrap().pages, source_pages);

        // Critical pressure preserves pinned T0 pages.
        let release = cache
            .handle_memory_pressure(&mut pool, MemoryPressure::Critical)
            .unwrap();
        assert!(release.hot.is_empty());
        cache.release_attachment(&mut pool, &attached).unwrap();

        let release = cache
            .handle_memory_pressure(&mut pool, MemoryPressure::Critical)
            .unwrap();
        assert_eq!(release.hot.len(), 2);
        assert_eq!(pool.stats().used_pages, 0);
    }

    #[test]
    fn critical_purge_of_unpinned_hot_pages_unblocks_cache_miss_admission() {
        let mut pool = PagedKvPool::new(PagedKvConfig::apple_v1(3, 2)).unwrap();
        let source = pool.allocate_chain(ReqId(1), 65).unwrap();
        let mut cache = PagedPromptCache::new(config(), LAYOUT, 16).unwrap();
        cache
            .promote_hot(&mut pool, identity(), &prompt(), source)
            .unwrap();
        pool.release_chain(source).unwrap();
        assert_eq!(pool.stats().free_pages, 1);
        assert!(matches!(
            pool.allocate_chain(ReqId(2), 65),
            Err(PagedKvError::PagePoolExhausted { .. })
        ));

        cache
            .handle_memory_pressure(&mut pool, MemoryPressure::Critical)
            .unwrap();
        assert_eq!(pool.stats().free_pages, 3);
        let miss = pool.allocate_chain(ReqId(2), 65).unwrap();
        assert_eq!(pool.view(miss).unwrap().token_len, 65);
    }

    #[test]
    fn warm_capture_and_restore_moves_exact_bytes_into_private_pages() {
        let mut pool = PagedKvPool::new(PagedKvConfig::apple_v1(6, 3)).unwrap();
        let source = pool.allocate_chain(ReqId(1), 65).unwrap();
        let source_pages = pool.view(source).unwrap().pages[..2].to_vec();
        let mut io = FakeIo::new(6, 16);
        let expected: Vec<_> = source_pages
            .iter()
            .map(|page| io.pages[page.0 as usize].clone())
            .collect();
        let mut cache = PagedPromptCache::new(config(), LAYOUT, 16).unwrap();
        let (admission, _) = cache
            .capture_warm(&pool, &mut io, identity(), &prompt(), source, || false)
            .unwrap();
        assert_eq!(admission.admitted_pages, 2);
        pool.release_chain(source).unwrap();

        let attached = cache
            .attach(&mut pool, &mut io, ReqId(2), &identity(), &prompt(), || {
                false
            })
            .unwrap()
            .unwrap();
        assert_eq!(attached.tier, AttachedCacheTier::T2Warm);
        let destinations = pool.view(attached.chain).unwrap().pages.to_vec();
        for (destination, bytes) in destinations.iter().zip(expected) {
            assert_eq!(io.pages[destination.0 as usize], bytes);
        }
        assert_eq!(io.restores, 2);
        cache.release_attachment(&mut pool, &attached).unwrap();
    }

    #[test]
    fn prepared_warm_restore_survives_eviction_and_busy_materialization() {
        let mut pool = PagedKvPool::new(PagedKvConfig::apple_v1(6, 3)).unwrap();
        let source = pool.allocate_chain(ReqId(1), 65).unwrap();
        let source_pages = pool.view(source).unwrap().pages[..2].to_vec();
        let mut io = FakeIo::new(6, 16);
        let expected: Vec<_> = source_pages
            .iter()
            .map(|page| io.pages[page.0 as usize].clone())
            .collect();
        let mut cache = PagedPromptCache::new(config(), LAYOUT, 16).unwrap();
        cache
            .capture_warm(&pool, &mut io, identity(), &prompt(), source, || false)
            .unwrap();
        pool.release_chain(source).unwrap();

        let prepared = cache
            .prepare_warm_restore(&identity(), &prompt())
            .unwrap()
            .unwrap();
        assert_eq!(prepared.matched_tokens(), 64);
        assert_eq!(prepared.page_count(), 2);

        // The prepared value owns immutable bytes independently of T2
        // metadata, including across memory-pressure eviction.
        cache
            .handle_memory_pressure(&mut pool, MemoryPressure::Critical)
            .unwrap();
        assert!(cache
            .prepare_warm_restore(&identity(), &prompt())
            .unwrap()
            .is_none());

        io.restore_error = Some(KvPageIoError::BackendBusy);
        assert!(matches!(
            cache.materialize_prepared_warm(&mut pool, &mut io, ReqId(2), &prepared, || false),
            Err(PagedPromptCacheError::PageIo(KvPageIoError::BackendBusy))
        ));
        assert_eq!(pool.chain_for_owner(ReqId(2)), None);
        assert_eq!(pool.stats().used_pages, 0);

        // Busy is non-destructive: the same prepared ownership can be retried
        // after the accelerator reaches its idle page-I/O barrier.
        io.restore_error = None;
        let attached = cache
            .materialize_prepared_warm(&mut pool, &mut io, ReqId(2), &prepared, || false)
            .unwrap()
            .unwrap();
        let destinations = pool.view(attached.chain).unwrap().pages.to_vec();
        for (destination, bytes) in destinations.iter().zip(expected) {
            assert_eq!(io.pages[destination.0 as usize], bytes);
        }
        cache.release_attachment(&mut pool, &attached).unwrap();
        assert_eq!(pool.stats().used_pages, 0);
    }

    #[test]
    fn prepared_warm_restore_cancellation_releases_transactional_chain() {
        let mut pool = PagedKvPool::new(PagedKvConfig::apple_v1(4, 2)).unwrap();
        let mut cache = PagedPromptCache::new(config(), LAYOUT, 8).unwrap();
        cache
            .metadata
            .insert_warm(
                identity(),
                &prompt(),
                vec![
                    WarmPage {
                        bytes: vec![1; 8].into(),
                    },
                    WarmPage {
                        bytes: vec![2; 8].into(),
                    },
                ],
            )
            .unwrap();
        let prepared = cache
            .prepare_warm_restore(&identity(), &prompt())
            .unwrap()
            .unwrap();
        let mut io = FakeIo::new(4, 8);
        let polls = Cell::new(0);
        assert!(matches!(
            cache.materialize_prepared_warm(&mut pool, &mut io, ReqId(3), &prepared, || {
                let next = polls.get() + 1;
                polls.set(next);
                next >= 3
            }),
            Err(PagedPromptCacheError::Cancelled)
        ));
        assert_eq!(pool.chain_for_owner(ReqId(3)), None);
        assert_eq!(pool.stats().used_pages, 0);
    }

    #[test]
    fn prepared_persistent_restore_does_no_io_until_transactional_materialization() {
        let mut pool = PagedKvPool::new(PagedKvConfig::apple_v1(4, 2)).unwrap();
        let mut cache = PagedPromptCache::new(config(), LAYOUT, 8).unwrap();
        let record = PersistentCacheRecord {
            matched_tokens: 64,
            pages: vec![
                WarmPage {
                    bytes: vec![1; 8].into(),
                },
                WarmPage {
                    bytes: vec![2; 8].into(),
                },
            ],
        };
        let prepared = cache
            .prepare_persistent_restore(&identity(), &prompt(), &record)
            .unwrap();
        assert_eq!(prepared.page_count(), 2);
        assert_eq!(pool.stats().used_pages, 0);

        let mut io = FakeIo::new(4, 8);
        io.restore_error = Some(KvPageIoError::BackendBusy);
        assert!(matches!(
            cache.materialize_prepared_warm(&mut pool, &mut io, ReqId(9), &prepared, || false),
            Err(PagedPromptCacheError::PageIo(KvPageIoError::BackendBusy))
        ));
        assert_eq!(pool.chain_for_owner(ReqId(9)), None);
        assert_eq!(pool.stats().used_pages, 0);
    }

    #[test]
    fn unavailable_backend_and_cancelled_restore_leave_no_chain() {
        let mut pool = PagedKvPool::new(PagedKvConfig::apple_v1(4, 2)).unwrap();
        let mut cache = PagedPromptCache::new(config(), LAYOUT, 8).unwrap();
        let pages = vec![
            WarmPage {
                bytes: vec![1; 8].into(),
            },
            WarmPage {
                bytes: vec![2; 8].into(),
            },
        ];
        cache
            .metadata
            .insert_warm(identity(), &prompt(), pages)
            .unwrap();

        let mut unavailable = UnavailableKvPageIo;
        assert!(matches!(
            cache.attach(
                &mut pool,
                &mut unavailable,
                ReqId(3),
                &identity(),
                &prompt(),
                || false
            ),
            Err(PagedPromptCacheError::PageIo(
                KvPageIoError::BackendUnavailable
            ))
        ));
        assert_eq!(pool.chain_for_owner(ReqId(3)), None);

        let mut io = FakeIo::new(4, 8);
        let polls = Cell::new(0);
        assert!(matches!(
            cache.attach(&mut pool, &mut io, ReqId(4), &identity(), &prompt(), || {
                let next = polls.get() + 1;
                polls.set(next);
                next >= 3
            }),
            Err(PagedPromptCacheError::Cancelled)
        ));
        assert_eq!(pool.chain_for_owner(ReqId(4)), None);
        assert_eq!(pool.stats().used_pages, 0);
    }

    #[test]
    fn identity_layout_mismatch_is_rejected_before_lookup_or_capture() {
        let mut wrong = identity();
        wrong.kv_layout = [9; 32];
        let mut cache = PagedPromptCache::new(config(), LAYOUT, 8).unwrap();
        let mut pool = PagedKvPool::new(PagedKvConfig::apple_v1(2, 1)).unwrap();
        let mut io = UnavailableKvPageIo;
        assert!(matches!(
            cache.attach(&mut pool, &mut io, ReqId(1), &wrong, &prompt(), || false),
            Err(PagedPromptCacheError::IdentityLayoutMismatch)
        ));
    }

    #[test]
    fn cow_tail_fails_closed_without_copy_hook_then_copies_exact_page() {
        let mut pool = PagedKvPool::new(PagedKvConfig::apple_v1(4, 3)).unwrap();
        let source = pool.allocate_chain(ReqId(1), 17).unwrap();
        let fork = pool.fork_prefix(source, ReqId(2), 17).unwrap();
        let original = pool.view(fork).unwrap().pages[0];
        assert_eq!(pool.page_ref_count(original), Some(2));
        let before = pool.stats();
        let mut cache = PagedPromptCache::new(config(), LAYOUT, 8).unwrap();
        let mut unavailable = UnavailableKvPageIo;
        assert!(matches!(
            cache.make_tail_writable(&mut pool, &mut unavailable, fork),
            Err(PagedPromptCacheError::PageIo(
                KvPageIoError::BackendUnavailable
            ))
        ));
        assert_eq!(pool.stats(), before);
        assert_eq!(pool.view(fork).unwrap().pages[0], original);

        let mut io = FakeIo::new(4, 8);
        let expected = io.pages[original.0 as usize].clone();
        let result = cache.make_tail_writable(&mut pool, &mut io, fork).unwrap();
        let destination = match result {
            CowTailResult::Copied { destination, .. } => destination,
            other => panic!("expected physical COW, got {other:?}"),
        };
        assert_ne!(destination, original);
        assert_eq!(io.pages[destination.0 as usize], expected);
        assert_eq!(io.copies, 1);
        assert_eq!(pool.page_ref_count(original), Some(1));
        assert_eq!(pool.page_ref_count(destination), Some(1));
    }

    #[test]
    fn warm_pages_promote_to_leased_hot_pages_without_double_accounting() {
        let mut pool = PagedKvPool::new(PagedKvConfig::apple_v1(4, 2)).unwrap();
        let source = pool.allocate_chain(ReqId(1), 65).unwrap();
        let mut io = FakeIo::new(4, 16);
        let mut cache = PagedPromptCache::new(config(), LAYOUT, 16).unwrap();
        cache
            .capture_warm(&pool, &mut io, identity(), &prompt(), source, || false)
            .unwrap();
        assert_eq!(cache.metadata().metrics().warm_bytes, 32);
        let promoted = cache
            .promote_hot(&mut pool, identity(), &prompt(), source)
            .unwrap();
        assert_eq!(promoted.admitted_pages, 2);
        assert_eq!(promoted.evicted_warm_bytes, 32);
        assert_eq!(cache.metadata().metrics().warm_bytes, 0);
        assert_eq!(cache.metadata().metrics().hot_bytes, 32);
        assert!(matches!(
            cache.metadata.lookup(&identity(), &prompt()),
            CacheLookup::Hit(hit) if hit.lowest_tier == PageResidency::Hot
        ));
    }

    #[test]
    fn persistent_record_restore_revalidates_reusable_length() {
        let mut pool = PagedKvPool::new(PagedKvConfig::apple_v1(4, 2)).unwrap();
        let mut io = FakeIo::new(4, 8);
        let mut cache = PagedPromptCache::new(config(), LAYOUT, 8).unwrap();
        let mut record = PersistentCacheRecord {
            matched_tokens: 64,
            pages: vec![
                WarmPage {
                    bytes: vec![1; 8].into(),
                },
                WarmPage {
                    bytes: vec![2; 8].into(),
                },
            ],
        };
        let attachment = cache
            .restore_persistent_record(
                &mut pool,
                &mut io,
                ReqId(1),
                &identity(),
                &prompt(),
                &record,
                || false,
            )
            .unwrap()
            .unwrap();
        cache.release_attachment(&mut pool, &attachment).unwrap();

        record.matched_tokens = 32;
        assert!(matches!(
            cache.restore_persistent_record(
                &mut pool,
                &mut io,
                ReqId(2),
                &identity(),
                &prompt(),
                &record,
                || false
            ),
            Err(PagedPromptCacheError::Metadata(
                CacheError::PageCountMismatch
            ))
        ));
        assert_eq!(pool.chain_for_owner(ReqId(2)), None);
    }
}
