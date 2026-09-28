// Offline Metal operator arm. No model state, queue daemon, or production hooks.
// All library/pipeline API calls happen in prepare(), before measured intervals.
import Foundation
#if canImport(Metal)
import Metal
import CryptoKit

struct BufferSpec: Codable { let id: Int; let file: String; let readOnly: Bool }
struct Binding: Codable { let index: Int; let id: Int; let offset: Int }
struct Constant: Codable { let index: Int; let word: UInt32 }
struct Pass: Codable {
    let kernel: String; let grid: [Int]; let threads: [Int]
    let bindings: [Binding]; let constants: [Constant]; let sourceSharedBytes: Int
}
struct Job: Codable {
    let schema: String; let library: String; let buffers: [BufferSpec]
    let sourceBodySha256: String
    let preflight: [Pass]; let passes: [Pass]; let warmup: Int; let repeats: Int
}
enum Failure: Error { case invalid(String) }
func need(_ test: Bool, _ message: String) throws {
    if !test { throw Failure.invalid(message) }
}
func sha(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
func size(_ a: [Int]) throws -> MTLSize {
    try need(a.count == 3 && a.allSatisfy { $0 > 0 && $0 <= 262144 }, "invalid dispatch dimensions")
    return MTLSize(width: a[0], height: a[1], depth: a[2])
}
final class Arm {
    let device: MTLDevice; let queue: MTLCommandQueue; let job: Job
    var buffers: [Int: MTLBuffer] = [:]; var original: [Int: Data] = [:]
    var states: [String: MTLComputePipelineState] = [:]
    var libraryLoads = 0; var pipelineCalls = 0
    var resources: [[String: Any]] = []
    init(_ job: Job) throws {
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue() else {
            throw Failure.invalid("Metal device/queue unavailable")
        }
        self.device = d; self.queue = q; self.job = job
        try need(job.schema == "rvllm.prefill-next.driver.v1", "wrong schema")
        try need(!job.passes.isEmpty && job.passes.count <= 8 && job.preflight.count == (needsLayout(job) ? 2 : 1),
                 "one bounded operator and one source identity preflight required")
        try need((0...20).contains(job.warmup) && (1...31).contains(job.repeats), "unbounded repeats")
    }
    func prepare() throws {
        // Counters wrap the actual API calls made by this driver. A library load
        // is NOT described as a source compilation. Metal's internal compilation
        // is not exposed by these counters.
        libraryLoads += 1
        let lib = try device.makeLibrary(URL: URL(fileURLWithPath: job.library))
        for p in job.preflight + job.passes where states[p.kernel] == nil {
            guard let fn = lib.makeFunction(name: p.kernel) else { throw Failure.invalid("missing function \(p.kernel)") }
            pipelineCalls += 1
            let state = try device.makeComputePipelineState(function: fn)
            try need(state.threadExecutionWidth == 32, "unexpected SIMD width")
            try need(state.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength,
                     "compiler-reported shared memory exceeds device")
            states[p.kernel] = state
            resources.append(["kernel":p.kernel,"execution_width":state.threadExecutionWidth,
                "max_threads":state.maxTotalThreadsPerThreadgroup,
                "static_shared_bytes":state.staticThreadgroupMemoryLength,
                "device_shared_bytes":device.maxThreadgroupMemoryLength])
        }
        try need(job.buffers.count <= 32, "too many buffers")
        for b in job.buffers {
            try need(buffers[b.id] == nil, "duplicate buffer id")
            let data = try Data(contentsOf: URL(fileURLWithPath: b.file))
            try need(data.count >= 65 && data.count <= 536870912, "buffer size outside bounds")
            let buffer: MTLBuffer? = data.withUnsafeBytes { raw in
                guard let address = raw.baseAddress else { return nil }
                return device.makeBuffer(bytes: address, length: data.count, options: .storageModeShared)
            }
            guard let buffer = buffer else { throw Failure.invalid("buffer allocation failed") }
            buffer.label = "prefill-fixture-\(b.id)"; buffers[b.id] = buffer; original[b.id] = data
        }
        for p in job.preflight + job.passes {
            _ = try size(p.grid); _ = try size(p.threads)
            guard let state = states[p.kernel] else { throw Failure.invalid("missing PSO") }
            try need(p.threads.reduce(1, *) <= state.maxTotalThreadsPerThreadgroup, "thread limit")
            try need(p.sourceSharedBytes >= 0 && p.sourceSharedBytes <= device.maxThreadgroupMemoryLength, "source shared budget")
            var indices = Set<Int>()
            for b in p.bindings {
                guard let buffer = buffers[b.id] else { throw Failure.invalid("unknown buffer") }
                try need(b.index >= 0 && b.index <= 30 && indices.insert(b.index).inserted,
                         "duplicate/invalid binding")
                try need(b.offset >= 32 && b.offset < buffer.length - 32 && b.offset % 2 == 0, "binding offset")
            }
            for c in p.constants {
                try need(c.index >= 0 && c.index <= 30 && indices.insert(c.index).inserted,
                         "duplicate/invalid constant")
            }
        }
    }
    func execute(_ passes: [Pass]) throws -> [String: Any] {
        let beforeLib = libraryLoads, beforePSO = pipelineCalls
        let start = DispatchTime.now().uptimeNanoseconds
        guard let cb = queue.makeCommandBuffer() else { throw Failure.invalid("command buffer") }
        for p in passes {
            guard let enc = cb.makeComputeCommandEncoder(), let state = states[p.kernel] else {
                throw Failure.invalid("encoder/PSO")
            }
            enc.label = p.kernel; enc.setComputePipelineState(state)
            for b in p.bindings { enc.setBuffer(buffers[b.id]!, offset: b.offset, index: b.index) }
            for c in p.constants {
                var word = c.word.littleEndian
                enc.setBytes(&word, length: 4, index: c.index)
            }
            enc.dispatchThreadgroups(try size(p.grid), threadsPerThreadgroup: try size(p.threads))
            enc.endEncoding()
        }
        let encoded = DispatchTime.now().uptimeNanoseconds
        cb.commit(); cb.waitUntilCompleted()
        let completed = DispatchTime.now().uptimeNanoseconds
        try need(cb.status == .completed && cb.error == nil, "Metal command failed: \(String(describing: cb.error))")
        try need(beforeLib == libraryLoads && beforePSO == pipelineCalls, "inference-time pipeline/library API call")
        let gpu = cb.gpuEndTime - cb.gpuStartTime
        return ["cpu_encode_ns":encoded-start,"wait_ns":completed-encoded,"wall_ns":completed-start,
                "gpu_ns":gpu.isFinite && gpu > 0 ? (gpu * 1e9) as Any : NSNull(),
                "gpu_timer_valid":gpu.isFinite && gpu > 0,"dispatches":passes.map { $0.kernel },
                "library_load_calls":libraryLoads-beforeLib,"source_compile_calls":0,
                "pipeline_creation_calls":pipelineCalls-beforePSO,"completed":true]
    }
    func readbacks(_ directory: URL, _ suffix: String) throws -> [[String: Any]] {
        var result: [[String: Any]] = []
        for spec in job.buffers {
            let buffer = buffers[spec.id]!
            let bytes = Data(bytes: buffer.contents(), count: buffer.length)
            let initial = original[spec.id]!
            let guardOK = bytes.prefix(32) == initial.prefix(32) && bytes.suffix(32) == initial.suffix(32)
            let same = bytes == initial
            if !spec.readOnly { try bytes.write(to: directory.appendingPathComponent("buffer-\(spec.id)-\(suffix).bin"), options:.atomic) }
            result.append(["id":spec.id,"sha256":sha(bytes),"bytes":bytes.count,
                           "guards_ok":guardOK,"read_only":spec.readOnly,"unchanged":same])
            // Preserve every mutable buffer even on a guard/input failure.
            // The parent referee checks these flags before accepting any cell.
        }
        return result
    }
}
// A new candidate cannot omit its same-process fragment-ABI gate.
func needsLayout(_ job: Job) -> Bool {
    job.passes.contains { $0.kernel.hasPrefix("research_prefill27_") || $0.kernel == "pr27_fragment_layout_probe" }
}
func verifyLayout(_ data: Data) throws {
    try need(data.count == 576, "fragment layout buffer extent")
    try need(data.prefix(32).allSatisfy { $0 == 0xa5 } && data.suffix(32).allSatisfy { $0 == 0xa5 }, "fragment probe guards")
    let bytes = [UInt8](data[32..<544])
    for i in 0..<128 {
        let at = i * 4
        let bits = UInt32(bytes[at]) | UInt32(bytes[at+1]) << 8 | UInt32(bytes[at+2]) << 16 | UInt32(bytes[at+3]) << 24
        let got = Float(bitPattern: bits)
        var expected: Float = 0
        if i < 64 { expected = Float(i + 1) }
        else {
            let r = (i - 64) / 8, c = (i - 64) % 8
            for k in 0..<8 { expected += Float((r*8+k+1)*(k+1)*(c+2))/16 }
        }
        try need(got.isFinite && got.bitPattern == expected.bitPattern, "fragment layout/MMA ABI mismatch at \(i)")
    }
}
func conditions() -> [String: Any] {
    let p = ProcessInfo.processInfo
    var result: [String: Any] = ["thermal_state": p.thermalState.rawValue,
        "active_processors": p.activeProcessorCount,
        "observed_only": true, "AC_power_observed": false]
    if #available(macOS 12.0, *) { result["low_power_mode"] = p.isLowPowerModeEnabled }
    return result
}
func run() throws {
    try need(CommandLine.arguments.count == 3, "usage: MetalArm JOB.json OUTPUT_DIR")
    let jobPath = URL(fileURLWithPath:CommandLine.arguments[1])
    let output = URL(fileURLWithPath:CommandLine.arguments[2], isDirectory:true)
    try FileManager.default.createDirectory(at:output,withIntermediateDirectories:true)
    let job = try JSONDecoder().decode(Job.self,from:Data(contentsOf:jobPath))
    let arm = try Arm(job); try arm.prepare()
    let beforeConditions = conditions()
    let preflight = try arm.execute([job.preflight[0]])
    // Authenticate the loaded library before executing any workload kernel.
    guard let identityBuffer = arm.buffers[31] else { throw Failure.invalid("missing identity buffer") }
    let identity = Data(bytes: identityBuffer.contents(), count: identityBuffer.length)
    try need(identity.count == 96, "identity extent")
    let identityBytes = [UInt8](identity[32..<64])
    var observed = ""
    for i in stride(from: 0, to: 32, by: 4) {
        let word = UInt32(identityBytes[i]) | UInt32(identityBytes[i+1]) << 8
            | UInt32(identityBytes[i+2]) << 16 | UInt32(identityBytes[i+3]) << 24
        observed += String(format: "%08x", word)
    }
    try need(observed == job.sourceBodySha256, "loaded library source identity mismatch")
    var layout: [String: Any] = ["required":false,"verified":false]
    if needsLayout(job) {
        let p = job.preflight[1]
        try need(p.kernel == "pr27_fragment_layout_probe" && p.grid == [1,1,1] && p.threads == [32,1,1]
            && p.bindings.count == 1 && p.bindings[0].id == 30 && p.bindings[0].index == 0
            && p.bindings[0].offset == 32 && p.constants.isEmpty, "wrong fragment preflight")
        layout = try arm.execute([p])
        guard let probe = arm.buffers[30] else { throw Failure.invalid("layout buffer missing") }
        let bytes = Data(bytes:probe.contents(), count:probe.length)
        _ = try arm.readbacks(output,"preflight") // retain bytes even on rejection
        try verifyLayout(bytes)
        layout["required"] = true; layout["verified"] = true
    }
    let first = try arm.execute(job.passes)
    let initial = try arm.readbacks(output,"first")
    for _ in 0..<job.warmup { _ = try arm.execute(job.passes) }
    var samples: [[String: Any]] = []
    for _ in 0..<job.repeats { samples.append(try arm.execute(job.passes)) }
    let final = try arm.readbacks(output,"last")
    let report: [String: Any] = ["schema":"rvllm.prefill-next.driver-receipt.v1",
        "device":arm.device.name,"registry_id":String(arm.device.registryID),
        "apple9":arm.device.supportsFamily(.apple9),
        "os":ProcessInfo.processInfo.operatingSystemVersionString,
        "library_load_calls_prepare":arm.libraryLoads,"pipeline_creation_calls_prepare":arm.pipelineCalls,
        "preflight":preflight,"fragment_layout":layout,"source_identity_verified_before_operator":true,"correctness":first,"resources":arm.resources,
        "first_readbacks":initial,"last_readbacks":final,"samples":samples,
        "conditions":["before":beforeConditions,"after":conditions()],
        "production_promotion":false]
    try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys])
        .write(to:output.appendingPathComponent("driver.json"),options:.atomic)
}
do { try run() } catch {
    FileHandle.standardError.write(Data("MetalArm error: \(error)\n".utf8)); exit(1)
}
#else
FileHandle.standardError.write(Data("MetalArm requires Apple Metal; no fallback is provided.\n".utf8))
exit(2)
#endif
