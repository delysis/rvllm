//! Exact, tenant-isolated, page-aligned prompt-prefix cache metadata.
//!
//! The cache deliberately separates active pins (T0), hot accelerator page
//! handles (T1), and bit-exact warm bytes (T2). T3 persistence belongs to the
//! platform host because encryption keys and iOS Data Protection are host
//! lifecycle concerns.

use rvllm_core::TokenId;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

pub const PROMPT_CACHE_PAGE_TOKENS: usize = 32;

#[must_use]
pub const fn reusable_prompt_tokens(prompt_len: usize) -> usize {
    (prompt_len.saturating_sub(1) / PROMPT_CACHE_PAGE_TOKENS) * PROMPT_CACHE_PAGE_TOKENS
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct CacheNamespace(String);

impl CacheNamespace {
    pub fn new(value: impl Into<String>) -> Result<Self, CacheError> {
        let value = value.into();
        if value.is_empty() || value.len() > 128 {
            return Err(CacheError::InvalidNamespace);
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct CacheIdentity {
    pub namespace: CacheNamespace,
    pub model: [u8; 32],
    pub tokenizer: [u8; 32],
    pub adapter: Option<[u8; 32]>,
    pub kv_layout: [u8; 32],
    pub numeric_path: [u8; 32],
    pub format_version: u32,
}

impl CacheIdentity {
    #[must_use]
    pub fn fingerprint(&self) -> CacheFingerprint {
        let mut h = Sha256::new();
        h.update(b"rvllm.prompt-cache.identity.v1");
        h.update((self.namespace.0.len() as u32).to_le_bytes());
        h.update(self.namespace.0.as_bytes());
        h.update(self.model);
        h.update(self.tokenizer);
        match self.adapter {
            Some(adapter) => {
                h.update([1]);
                h.update(adapter);
            }
            None => h.update([0]),
        }
        h.update(self.kv_layout);
        h.update(self.numeric_path);
        h.update(self.format_version.to_le_bytes());
        CacheFingerprint(h.finalize().into())
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct CacheFingerprint(pub [u8; 32]);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct HotPageHandle {
    pub id: u64,
    pub bytes: usize,
}

#[derive(Clone, Eq, PartialEq)]
pub struct WarmPage {
    pub bytes: Arc<[u8]>,
}

impl fmt::Debug for WarmPage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WarmPage")
            .field(
                "bytes",
                &format_args!("<redacted:{} bytes>", self.bytes.len()),
            )
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CachedPage {
    Hot(HotPageHandle),
    Warm(WarmPage),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PageResidency {
    Hot,
    Warm,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CacheMatch {
    pub match_id: u64,
    pub matched_tokens: usize,
    pub pages: Vec<CachedPage>,
    pub lowest_tier: PageResidency,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CacheLookup {
    Miss,
    Hit(CacheMatch),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PromptCacheConfig {
    pub hot_bytes: usize,
    pub warm_bytes: usize,
    pub protected_fraction_percent: u8,
    pub frequency_aging_interval: u64,
}

impl Default for PromptCacheConfig {
    fn default() -> Self {
        Self {
            hot_bytes: 256 * 1024 * 1024,
            warm_bytes: if cfg!(target_os = "ios") {
                0
            } else {
                256 * 1024 * 1024
            },
            protected_fraction_percent: 80,
            frequency_aging_interval: 4096,
        }
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MemoryPressure {
    /// The host has explicitly reported that pressure has ended.
    Normal,
    Warning,
    Critical,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PromptCacheMetrics {
    pub lookups: u64,
    pub hot_hits: u64,
    pub warm_hits: u64,
    pub misses: u64,
    pub collision_candidates_examined: u64,
    pub exact_verification_failures: u64,
    pub tokens_reused: u64,
    pub admissions: u64,
    pub rejections: u64,
    pub hot_evictions: u64,
    pub warm_evictions: u64,
    pub pressure_purges: u64,
    pub hot_bytes: usize,
    pub warm_bytes: usize,
    pub peak_hot_bytes: usize,
    pub peak_warm_bytes: usize,
    pub pinned_pages: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CacheError {
    InvalidNamespace,
    InvalidConfig,
    PageCountMismatch,
    ZeroSizedPage,
    UnknownMatch,
    NotFullyHot,
    UnknownLease,
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidNamespace => write!(f, "invalid prompt-cache namespace"),
            Self::InvalidConfig => write!(f, "invalid prompt-cache configuration"),
            Self::PageCountMismatch => write!(f, "prompt-cache page count mismatch"),
            Self::ZeroSizedPage => write!(f, "prompt-cache page must not be empty"),
            Self::UnknownMatch => write!(f, "unknown prompt-cache match"),
            Self::NotFullyHot => write!(f, "prompt-cache match is not fully hot"),
            Self::UnknownLease => write!(f, "unknown prompt-cache pin lease"),
        }
    }
}

impl std::error::Error for CacheError {}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Admission {
    pub matched_tokens: usize,
    pub admitted_pages: usize,
    pub rejected: bool,
    pub evicted_hot: Vec<HotPageHandle>,
    pub evicted_warm_bytes: usize,
    pub unused_hot: Vec<HotPageHandle>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PressureRelease {
    pub hot: Vec<HotPageHandle>,
    pub warm_bytes: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Segment {
    Probation,
    Protected,
}

#[derive(Clone, Debug)]
enum Payload {
    Hot(HotPageHandle),
    Warm(WarmPage),
}

#[derive(Clone, Debug)]
struct Node {
    parent: Option<usize>,
    depth: usize,
    tokens: [TokenId; PROMPT_CACHE_PAGE_TOKENS],
    digest: CacheFingerprint,
    children: HashMap<CacheFingerprint, Vec<usize>>,
    payload: Option<Payload>,
    pins: u32,
    frequency: u32,
    last_touch: u64,
    segment: Segment,
}

type DigestFn = fn(&[u8]) -> [u8; 32];

pub struct PromptCache {
    config: PromptCacheConfig,
    roots: HashMap<CacheIdentity, HashMap<CacheFingerprint, Vec<usize>>>,
    nodes: Vec<Node>,
    frequency_sketch: HashMap<CacheFingerprint, u32>,
    leases: HashMap<u64, Vec<usize>>,
    next_lease: u64,
    clock: u64,
    digest_fn: DigestFn,
    metrics: PromptCacheMetrics,
}

impl PromptCache {
    pub fn new(config: PromptCacheConfig) -> Result<Self, CacheError> {
        if config.protected_fraction_percent > 100 || config.frequency_aging_interval == 0 {
            return Err(CacheError::InvalidConfig);
        }
        Ok(Self {
            config,
            roots: HashMap::new(),
            nodes: Vec::new(),
            frequency_sketch: HashMap::new(),
            leases: HashMap::new(),
            next_lease: 1,
            clock: 0,
            digest_fn: blake3_digest,
            metrics: PromptCacheMetrics::default(),
        })
    }

    #[cfg(test)]
    fn with_digest_fn(config: PromptCacheConfig, digest_fn: DigestFn) -> Self {
        let mut cache = Self::new(config).expect("valid config");
        cache.digest_fn = digest_fn;
        cache
    }

    #[must_use]
    pub fn metrics(&self) -> PromptCacheMetrics {
        self.metrics.clone()
    }

    pub fn lookup(&mut self, identity: &CacheIdentity, prompt: &[TokenId]) -> CacheLookup {
        self.metrics.lookups += 1;
        self.tick();
        let reusable = reusable_prompt_tokens(prompt.len());
        let mut parent = None;
        let mut previous = identity.fingerprint();
        let mut pages = Vec::new();
        let mut terminal = None;
        let mut lowest_tier = PageResidency::Hot;

        for page in prompt[..reusable].chunks_exact(PROMPT_CACHE_PAGE_TOKENS) {
            let exact: [TokenId; PROMPT_CACHE_PAGE_TOKENS] = page.try_into().unwrap();
            let digest = self.page_digest(previous, &exact);
            self.observe_digest(digest);
            let Some(node_id) = self.find_exact(identity, parent, digest, &exact) else {
                break;
            };
            let Some(payload) = self.nodes[node_id].payload.clone() else {
                break;
            };
            match payload {
                Payload::Hot(page) => pages.push(CachedPage::Hot(page)),
                Payload::Warm(page) => {
                    pages.push(CachedPage::Warm(page));
                    lowest_tier = PageResidency::Warm;
                }
            }
            self.touch(node_id);
            terminal = Some(node_id);
            parent = Some(node_id);
            previous = digest;
        }

        self.enforce_protected_fraction();
        let Some(terminal) = terminal else {
            self.metrics.misses += 1;
            return CacheLookup::Miss;
        };
        let matched_tokens = pages.len() * PROMPT_CACHE_PAGE_TOKENS;
        match lowest_tier {
            PageResidency::Hot => self.metrics.hot_hits += 1,
            PageResidency::Warm => self.metrics.warm_hits += 1,
        }
        self.metrics.tokens_reused += matched_tokens as u64;
        CacheLookup::Hit(CacheMatch {
            match_id: terminal as u64 + 1,
            matched_tokens,
            pages,
            lowest_tier,
        })
    }

    pub fn insert_hot(
        &mut self,
        identity: CacheIdentity,
        prompt: &[TokenId],
        handles: Vec<HotPageHandle>,
    ) -> Result<Admission, CacheError> {
        if handles.iter().any(|page| page.bytes == 0) {
            return Err(CacheError::ZeroSizedPage);
        }
        let reusable = reusable_prompt_tokens(prompt.len());
        if handles.len() != reusable / PROMPT_CACHE_PAGE_TOKENS {
            return Err(CacheError::PageCountMismatch);
        }
        self.tick();
        let node_ids = self.ensure_path(&identity, &prompt[..reusable]);
        let needed: usize = node_ids
            .iter()
            .zip(&handles)
            .filter(|(id, _)| !matches!(self.nodes[**id].payload, Some(Payload::Hot(_))))
            .map(|(_, page)| page.bytes)
            .sum();
        let excluded: HashSet<_> = node_ids.iter().copied().collect();
        let incoming_frequency = node_ids
            .iter()
            .map(|&id| self.estimated_frequency(self.nodes[id].digest))
            .min()
            .unwrap_or(1)
            .max(1);
        let mut admission = Admission {
            matched_tokens: reusable,
            ..Admission::default()
        };
        if !self.make_hot_room(
            needed,
            incoming_frequency,
            &excluded,
            &mut admission.evicted_hot,
        ) {
            admission.rejected = true;
            admission.unused_hot = handles;
            self.metrics.rejections += 1;
            return Ok(admission);
        }
        for (node_id, handle) in node_ids.into_iter().zip(handles) {
            if matches!(self.nodes[node_id].payload, Some(Payload::Hot(_))) {
                admission.unused_hot.push(handle);
                continue;
            }
            if let Some(Payload::Warm(warm)) = self.nodes[node_id].payload.take() {
                let bytes = warm.bytes.len();
                self.metrics.warm_bytes -= bytes;
                admission.evicted_warm_bytes += bytes;
            }
            self.nodes[node_id].payload = Some(Payload::Hot(handle));
            self.nodes[node_id].frequency = self.nodes[node_id]
                .frequency
                .max(self.estimated_frequency(self.nodes[node_id].digest))
                .max(1);
            self.nodes[node_id].last_touch = self.clock;
            self.nodes[node_id].segment = Segment::Probation;
            self.metrics.hot_bytes += handle.bytes;
            admission.admitted_pages += 1;
        }
        if admission.admitted_pages > 0 {
            self.metrics.admissions += 1;
        }
        self.update_peaks();
        Ok(admission)
    }

    pub fn insert_warm(
        &mut self,
        identity: CacheIdentity,
        prompt: &[TokenId],
        pages: Vec<WarmPage>,
    ) -> Result<Admission, CacheError> {
        let reusable = reusable_prompt_tokens(prompt.len());
        if pages.len() != reusable / PROMPT_CACHE_PAGE_TOKENS {
            return Err(CacheError::PageCountMismatch);
        }
        self.tick();
        let node_ids = self.ensure_path(&identity, &prompt[..reusable]);
        let mut admission = Admission {
            matched_tokens: reusable,
            ..Admission::default()
        };
        for (node_id, page) in node_ids.into_iter().zip(pages) {
            if self.nodes[node_id].payload.is_some() {
                continue;
            }
            let bytes = page.bytes.len();
            if bytes == 0 {
                return Err(CacheError::ZeroSizedPage);
            }
            admission.evicted_warm_bytes += self.make_warm_room(bytes);
            if self.metrics.warm_bytes + bytes > self.config.warm_bytes {
                admission.rejected = true;
                self.metrics.rejections += 1;
                continue;
            }
            self.nodes[node_id].payload = Some(Payload::Warm(page));
            self.nodes[node_id].frequency = self.nodes[node_id].frequency.max(1);
            self.nodes[node_id].last_touch = self.clock;
            self.metrics.warm_bytes += bytes;
            admission.admitted_pages += 1;
        }
        if admission.admitted_pages > 0 {
            self.metrics.admissions += 1;
        }
        self.update_peaks();
        Ok(admission)
    }

    pub fn pin(&mut self, match_id: u64) -> Result<u64, CacheError> {
        let id = self.node_id(match_id)?;
        let chain = self.chain(id);
        if chain
            .iter()
            .any(|&node| !matches!(self.nodes[node].payload, Some(Payload::Hot(_))))
        {
            return Err(CacheError::NotFullyHot);
        }
        for &node in &chain {
            if self.nodes[node].pins == 0 {
                self.metrics.pinned_pages += 1;
            }
            self.nodes[node].pins += 1;
        }
        let lease = self.next_lease;
        self.next_lease += 1;
        self.leases.insert(lease, chain);
        Ok(lease)
    }

    pub fn release(&mut self, lease: u64) -> Result<(), CacheError> {
        let chain = self.leases.remove(&lease).ok_or(CacheError::UnknownLease)?;
        for node in chain {
            self.nodes[node].pins -= 1;
            if self.nodes[node].pins == 0 {
                self.metrics.pinned_pages -= 1;
            }
        }
        Ok(())
    }

    pub fn invalidate_identity(&mut self, identity: &CacheIdentity) -> PressureRelease {
        let mut release = PressureRelease::default();
        for id in self.identity_nodes(identity) {
            self.purge(id, &mut release);
        }
        self.roots.remove(identity);
        release
    }

    pub fn invalidate_namespace(&mut self, namespace: &CacheNamespace) -> PressureRelease {
        let identities: Vec<_> = self
            .roots
            .keys()
            .filter(|id| &id.namespace == namespace)
            .cloned()
            .collect();
        let mut release = PressureRelease::default();
        for identity in identities {
            let one = self.invalidate_identity(&identity);
            release.hot.extend(one.hot);
            release.warm_bytes += one.warm_bytes;
        }
        release
    }

    pub fn handle_memory_pressure(&mut self, pressure: MemoryPressure) -> PressureRelease {
        if pressure == MemoryPressure::Normal {
            return PressureRelease::default();
        }
        let mut release = PressureRelease::default();
        let ids: Vec<_> = (0..self.nodes.len()).collect();
        for &id in &ids {
            if self.nodes[id].pins == 0 && matches!(self.nodes[id].payload, Some(Payload::Warm(_)))
            {
                self.purge(id, &mut release);
            }
        }
        let hot_target = match pressure {
            MemoryPressure::Normal => unreachable!("normal pressure returned above"),
            MemoryPressure::Warning => self.config.hot_bytes / 2,
            MemoryPressure::Critical => 0,
        };
        let mut candidates: Vec<_> = ids
            .into_iter()
            .filter(|&id| {
                self.nodes[id].pins == 0 && matches!(self.nodes[id].payload, Some(Payload::Hot(_)))
            })
            .collect();
        candidates.sort_by_key(|&id| {
            let protected = usize::from(self.nodes[id].segment == Segment::Protected);
            (protected, self.nodes[id].last_touch, id)
        });
        for id in candidates {
            if self.metrics.hot_bytes <= hot_target {
                break;
            }
            self.purge(id, &mut release);
        }
        self.metrics.pressure_purges +=
            release.hot.len() as u64 + u64::from(release.warm_bytes > 0);
        release
    }

    fn ensure_path(&mut self, identity: &CacheIdentity, tokens: &[TokenId]) -> Vec<usize> {
        let mut parent = None;
        let mut previous = identity.fingerprint();
        let mut result = Vec::new();
        for page in tokens.chunks_exact(PROMPT_CACHE_PAGE_TOKENS) {
            let exact: [TokenId; PROMPT_CACHE_PAGE_TOKENS] = page.try_into().unwrap();
            let digest = self.page_digest(previous, &exact);
            let id = self
                .find_exact(identity, parent, digest, &exact)
                .unwrap_or_else(|| {
                    let id = self.nodes.len();
                    self.nodes.push(Node {
                        parent,
                        depth: result.len() + 1,
                        tokens: exact,
                        digest,
                        children: HashMap::new(),
                        payload: None,
                        pins: 0,
                        frequency: 0,
                        last_touch: self.clock,
                        segment: Segment::Probation,
                    });
                    if let Some(parent) = parent {
                        self.nodes[parent]
                            .children
                            .entry(digest)
                            .or_default()
                            .push(id);
                    } else {
                        self.roots
                            .entry(identity.clone())
                            .or_default()
                            .entry(digest)
                            .or_default()
                            .push(id);
                    }
                    id
                });
            result.push(id);
            parent = Some(id);
            previous = digest;
        }
        result
    }

    fn find_exact(
        &mut self,
        identity: &CacheIdentity,
        parent: Option<usize>,
        digest: CacheFingerprint,
        tokens: &[TokenId; PROMPT_CACHE_PAGE_TOKENS],
    ) -> Option<usize> {
        let bucket = match parent {
            Some(parent) => self.nodes[parent].children.get(&digest).cloned(),
            None => self
                .roots
                .get(identity)
                .and_then(|r| r.get(&digest))
                .cloned(),
        }?;
        for id in bucket {
            self.metrics.collision_candidates_examined += 1;
            if self.nodes[id].tokens == *tokens {
                return Some(id);
            }
            self.metrics.exact_verification_failures += 1;
        }
        None
    }

    fn page_digest(
        &self,
        previous: CacheFingerprint,
        page: &[TokenId; PROMPT_CACHE_PAGE_TOKENS],
    ) -> CacheFingerprint {
        let mut bytes = [0_u8; 32 + PROMPT_CACHE_PAGE_TOKENS * 4];
        bytes[..32].copy_from_slice(&previous.0);
        for (index, token) in page.iter().enumerate() {
            let offset = 32 + index * 4;
            bytes[offset..offset + 4].copy_from_slice(&token.raw().to_le_bytes());
        }
        CacheFingerprint((self.digest_fn)(&bytes))
    }

    fn touch(&mut self, id: usize) {
        self.nodes[id].frequency = self.nodes[id].frequency.saturating_add(1);
        self.nodes[id].last_touch = self.clock;
        self.nodes[id].segment = Segment::Protected;
    }

    fn tick(&mut self) {
        self.clock = self.clock.saturating_add(1);
        if self.clock % self.config.frequency_aging_interval == 0 {
            for node in &mut self.nodes {
                node.frequency /= 2;
            }
            self.frequency_sketch.retain(|_, frequency| {
                *frequency /= 2;
                *frequency > 0
            });
        }
    }

    fn observe_digest(&mut self, digest: CacheFingerprint) {
        let frequency = self.frequency_sketch.entry(digest).or_default();
        *frequency = frequency.saturating_add(1);
    }

    fn estimated_frequency(&self, digest: CacheFingerprint) -> u32 {
        self.frequency_sketch.get(&digest).copied().unwrap_or(0)
    }

    fn enforce_protected_fraction(&mut self) {
        let limit = self.config.hot_bytes * self.config.protected_fraction_percent as usize / 100;
        let mut protected: Vec<_> = (0..self.nodes.len())
            .filter(|&id| self.nodes[id].segment == Segment::Protected)
            .filter_map(|id| match self.nodes[id].payload {
                Some(Payload::Hot(page)) => Some((id, page.bytes)),
                _ => None,
            })
            .collect();
        let mut bytes: usize = protected.iter().map(|(_, bytes)| *bytes).sum();
        protected.sort_by_key(|(id, _)| (self.nodes[*id].last_touch, *id));
        for (id, page_bytes) in protected {
            if bytes <= limit {
                break;
            }
            self.nodes[id].segment = Segment::Probation;
            bytes -= page_bytes;
        }
    }

    fn make_hot_room(
        &mut self,
        needed: usize,
        incoming_frequency: u32,
        excluded: &HashSet<usize>,
        released: &mut Vec<HotPageHandle>,
    ) -> bool {
        if needed > self.config.hot_bytes {
            return false;
        }
        let required_release = self
            .metrics
            .hot_bytes
            .saturating_add(needed)
            .saturating_sub(self.config.hot_bytes);
        if required_release == 0 {
            return true;
        }
        let mut candidates: Vec<_> = (0..self.nodes.len())
            .filter(|id| !excluded.contains(id))
            .filter(|&id| {
                self.nodes[id].pins == 0 && matches!(self.nodes[id].payload, Some(Payload::Hot(_)))
            })
            .collect();
        candidates.sort_by_key(|&id| {
            let protected = usize::from(self.nodes[id].segment == Segment::Protected);
            (
                protected,
                self.nodes[id].frequency,
                self.nodes[id].last_touch,
                id,
            )
        });
        let mut victims = Vec::new();
        let mut release_bytes = 0usize;
        for victim in candidates {
            // TinyLFU admission: a cold candidate cannot displace an equally
            // or more frequently observed resident page.
            if incoming_frequency <= self.nodes[victim].frequency {
                return false;
            }
            let Some(Payload::Hot(page)) = self.nodes[victim].payload.as_ref() else {
                continue;
            };
            release_bytes = release_bytes.saturating_add(page.bytes);
            victims.push(victim);
            if release_bytes >= required_release {
                break;
            }
        }
        if release_bytes < required_release {
            return false;
        }
        for victim in victims {
            if let Some(Payload::Hot(page)) = self.nodes[victim].payload.take() {
                self.metrics.hot_bytes -= page.bytes;
                self.metrics.hot_evictions += 1;
                released.push(page);
            }
        }
        true
    }

    fn make_warm_room(&mut self, needed: usize) -> usize {
        let mut released = 0;
        while self.metrics.warm_bytes + needed > self.config.warm_bytes {
            let victim = (0..self.nodes.len())
                .filter(|&id| {
                    self.nodes[id].pins == 0
                        && matches!(self.nodes[id].payload, Some(Payload::Warm(_)))
                })
                .min_by_key(|&id| (self.nodes[id].frequency, self.nodes[id].last_touch, id));
            let Some(victim) = victim else {
                break;
            };
            if let Some(Payload::Warm(page)) = self.nodes[victim].payload.take() {
                let bytes = page.bytes.len();
                self.metrics.warm_bytes -= bytes;
                self.metrics.warm_evictions += 1;
                released += bytes;
            }
        }
        released
    }

    fn chain(&self, mut id: usize) -> Vec<usize> {
        let mut chain = Vec::with_capacity(self.nodes[id].depth);
        loop {
            chain.push(id);
            let Some(parent) = self.nodes[id].parent else {
                break;
            };
            id = parent;
        }
        chain.reverse();
        chain
    }

    fn node_id(&self, match_id: u64) -> Result<usize, CacheError> {
        let id = match_id.checked_sub(1).ok_or(CacheError::UnknownMatch)? as usize;
        (id < self.nodes.len())
            .then_some(id)
            .ok_or(CacheError::UnknownMatch)
    }

    fn identity_nodes(&self, identity: &CacheIdentity) -> Vec<usize> {
        let Some(roots) = self.roots.get(identity) else {
            return Vec::new();
        };
        let mut stack: Vec<_> = roots.values().flatten().copied().collect();
        let mut result = Vec::new();
        while let Some(id) = stack.pop() {
            result.push(id);
            stack.extend(self.nodes[id].children.values().flatten().copied());
        }
        result
    }

    fn purge(&mut self, id: usize, release: &mut PressureRelease) {
        if self.nodes[id].pins > 0 {
            return;
        }
        match self.nodes[id].payload.take() {
            Some(Payload::Hot(page)) => {
                self.metrics.hot_bytes -= page.bytes;
                release.hot.push(page);
            }
            Some(Payload::Warm(page)) => {
                self.metrics.warm_bytes -= page.bytes.len();
                release.warm_bytes += page.bytes.len();
            }
            None => {}
        }
    }

    fn update_peaks(&mut self) {
        self.metrics.peak_hot_bytes = self.metrics.peak_hot_bytes.max(self.metrics.hot_bytes);
        self.metrics.peak_warm_bytes = self.metrics.peak_warm_bytes.max(self.metrics.warm_bytes);
    }
}

fn blake3_digest(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(namespace: &str, marker: u8) -> CacheIdentity {
        CacheIdentity {
            namespace: CacheNamespace::new(namespace).unwrap(),
            model: [marker; 32],
            tokenizer: [2; 32],
            adapter: None,
            kv_layout: [3; 32],
            numeric_path: [4; 32],
            format_version: 1,
        }
    }

    fn tokens(count: usize, offset: u32) -> Vec<TokenId> {
        (0..count).map(|i| TokenId(offset + i as u32)).collect()
    }

    fn config() -> PromptCacheConfig {
        PromptCacheConfig {
            hot_bytes: 128,
            warm_bytes: 128,
            protected_fraction_percent: 80,
            frequency_aging_interval: 100,
        }
    }

    #[test]
    fn boundary_formula_leaves_final_token_private() {
        for len in 0..=32 {
            assert_eq!(reusable_prompt_tokens(len), 0);
        }
        for len in 33..=64 {
            assert_eq!(reusable_prompt_tokens(len), 32);
        }
        for len in 65..=96 {
            assert_eq!(reusable_prompt_tokens(len), 64);
        }
        assert_eq!(reusable_prompt_tokens(97), 96);
    }

    #[test]
    fn deepest_exact_prefix_hits_but_other_identity_misses() {
        let mut cache = PromptCache::new(config()).unwrap();
        let id = identity("tenant-a", 1);
        cache
            .insert_hot(
                id.clone(),
                &tokens(65, 0),
                vec![
                    HotPageHandle { id: 10, bytes: 16 },
                    HotPageHandle { id: 11, bytes: 16 },
                ],
            )
            .unwrap();
        let CacheLookup::Hit(hit) = cache.lookup(&id, &tokens(80, 0)) else {
            panic!("expected hit");
        };
        assert_eq!(hit.matched_tokens, 64);
        assert_eq!(
            cache.lookup(&identity("tenant-b", 1), &tokens(80, 0)),
            CacheLookup::Miss
        );
        assert_eq!(
            cache.lookup(&identity("tenant-a", 9), &tokens(80, 0)),
            CacheLookup::Miss
        );
    }

    #[test]
    fn forced_hash_collision_is_exactly_verified() {
        fn constant(_: &[u8]) -> [u8; 32] {
            [7; 32]
        }
        let mut cache = PromptCache::with_digest_fn(config(), constant);
        let id = identity("tenant", 1);
        cache
            .insert_hot(
                id.clone(),
                &tokens(33, 0),
                vec![HotPageHandle { id: 1, bytes: 16 }],
            )
            .unwrap();
        cache
            .insert_hot(
                id.clone(),
                &tokens(33, 1000),
                vec![HotPageHandle { id: 2, bytes: 16 }],
            )
            .unwrap();
        let CacheLookup::Hit(hit) = cache.lookup(&id, &tokens(40, 1000)) else {
            panic!("expected hit");
        };
        assert_eq!(
            hit.pages,
            vec![CachedPage::Hot(HotPageHandle { id: 2, bytes: 16 })]
        );
        assert!(cache.metrics().exact_verification_failures > 0);
        assert_eq!(cache.lookup(&id, &tokens(40, 2000)), CacheLookup::Miss);
    }

    #[test]
    fn critical_pressure_preserves_pins_then_purges_after_release() {
        let mut cache = PromptCache::new(config()).unwrap();
        let id = identity("tenant", 1);
        let prompt = tokens(33, 0);
        cache
            .insert_hot(
                id.clone(),
                &prompt,
                vec![HotPageHandle { id: 1, bytes: 64 }],
            )
            .unwrap();
        let CacheLookup::Hit(hit) = cache.lookup(&id, &prompt) else {
            panic!("expected hit");
        };
        let lease = cache.pin(hit.match_id).unwrap();
        assert!(cache
            .handle_memory_pressure(MemoryPressure::Critical)
            .hot
            .is_empty());
        cache.release(lease).unwrap();
        assert_eq!(
            cache
                .handle_memory_pressure(MemoryPressure::Critical)
                .hot
                .len(),
            1
        );
        assert_eq!(cache.metrics().hot_bytes, 0);
    }

    #[test]
    fn warning_purges_bit_exact_warm_pages() {
        let mut cache = PromptCache::new(config()).unwrap();
        let id = identity("tenant", 1);
        let bytes: Arc<[u8]> = vec![1, 2, 3, 4].into();
        cache
            .insert_warm(
                id.clone(),
                &tokens(33, 0),
                vec![WarmPage {
                    bytes: bytes.clone(),
                }],
            )
            .unwrap();
        let CacheLookup::Hit(hit) = cache.lookup(&id, &tokens(33, 0)) else {
            panic!("expected hit");
        };
        assert_eq!(hit.pages, vec![CachedPage::Warm(WarmPage { bytes })]);
        assert_eq!(
            cache
                .handle_memory_pressure(MemoryPressure::Warning)
                .warm_bytes,
            4
        );
        assert_eq!(cache.metrics().warm_bytes, 0);
    }

    #[test]
    fn normal_pressure_is_an_explicit_no_op() {
        assert_eq!(MemoryPressure::Normal as u8, 0);
        assert_eq!(MemoryPressure::Warning as u8, 1);
        assert_eq!(MemoryPressure::Critical as u8, 2);

        let mut cache = PromptCache::new(config()).unwrap();
        let id = identity("tenant", 1);
        cache
            .insert_hot(
                id.clone(),
                &tokens(33, 0),
                vec![HotPageHandle { id: 1, bytes: 64 }],
            )
            .unwrap();
        let release = cache.handle_memory_pressure(MemoryPressure::Normal);
        assert!(release.hot.is_empty());
        assert_eq!(release.warm_bytes, 0);
        assert_eq!(cache.metrics().hot_bytes, 64);
        assert!(matches!(
            cache.lookup(&id, &tokens(33, 0)),
            CacheLookup::Hit(_)
        ));
    }

    #[test]
    fn warm_page_debug_redacts_kv_bytes() {
        let page = WarmPage {
            bytes: Arc::from([0x13, 0x37, 0x42, 0x99]),
        };
        let rendered = format!("{page:?}");
        assert!(rendered.contains("<redacted:4 bytes>"));
        assert!(!rendered.contains("19"));
        assert!(!rendered.contains("55"));
        assert!(!rendered.contains("66"));
        assert!(!rendered.contains("153"));
    }

    #[test]
    fn tinylfu_rejects_a_cold_page_then_admits_it_after_repeated_misses() {
        let mut cache = PromptCache::new(config()).unwrap();
        let identity = identity("tenant-a", 1);
        let resident = tokens(33, 0);
        let candidate = tokens(33, 10_000);

        cache
            .insert_hot(
                identity.clone(),
                &resident,
                vec![HotPageHandle { id: 1, bytes: 128 }],
            )
            .unwrap();
        assert!(matches!(
            cache.lookup(&identity, &resident),
            CacheLookup::Hit(_)
        ));
        assert!(matches!(
            cache.lookup(&identity, &resident),
            CacheLookup::Hit(_)
        ));

        let cold = cache
            .insert_hot(
                identity.clone(),
                &candidate,
                vec![HotPageHandle { id: 2, bytes: 128 }],
            )
            .unwrap();
        assert!(cold.rejected);
        assert!(cold.evicted_hot.is_empty());

        for _ in 0..4 {
            assert_eq!(cache.lookup(&identity, &candidate), CacheLookup::Miss);
        }
        let admitted = cache
            .insert_hot(
                identity.clone(),
                &candidate,
                vec![HotPageHandle { id: 3, bytes: 128 }],
            )
            .unwrap();
        assert!(!admitted.rejected);
        assert_eq!(
            admitted.evicted_hot,
            vec![HotPageHandle { id: 1, bytes: 128 }]
        );
        assert!(matches!(
            cache.lookup(&identity, &candidate),
            CacheLookup::Hit(_)
        ));
    }
}
