#include "tensorlang/ExecutionEngine/JitRunner.h"
#include "tensorlang/Conversion/TensorLangToLinalg/TensorLangToLinalg.h"
#include "tensorlang/Dialect/TensorLang/Transforms/Passes.h"
#include "tensorlang/Dialect/TensorLang/IR/TensorLangDialect.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/Passes.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"

// Bufferization
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"

// Conversions
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Math/IR/Math.h"

using namespace mlir;
using namespace mlir::tensorlang;

//===----------------------------------------------------------------------===//
// JitRunner Implementation
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/EPCDynamicLibrarySearchGenerator.h"

// ... (existing includes)

JitRunner::JitRunner(std::unique_ptr<llvm::orc::LLJIT> jit) : jit(std::move(jit)) {}

JitRunner::~JitRunner() = default;

llvm::Expected<std::unique_ptr<JitRunner>> JitRunner::create() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto jitBuilder = llvm::orc::LLJITBuilder();
  auto jitOrError = jitBuilder.create();
  if (auto error = jitOrError.takeError())
    return std::move(error);

  auto jit = std::move(*jitOrError);
  
  jit->getMainJITDylib().addGenerator(
      cantFail(llvm::orc::EPCDynamicLibrarySearchGenerator::GetForTargetProcess(
          jit->getExecutionSession())));

  return std::unique_ptr<JitRunner>(new JitRunner(std::move(jit)));
}

llvm::Error JitRunner::run(ModuleOp module) {
  // 1. Run optimization pipeline: TensorLang -> Linalg -> LLVM
  PassManager pm(module.getContext());
  
  // Verify linearity
  pm.addPass(createVerifyLinearity());

  // Lower TensorLang to Linalg (tensors)
  pm.addPass(createConvertTensorLangToLinalgPass());

  // Bufferize (tensors -> memrefs)
  bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  pm.addPass(bufferization::createOneShotBufferizePass(options));

  // Lower Linalg -> SCF loops
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(memref::createExpandStridedMetadataPass());

  // Lower SCF -> ControlFlow
  pm.addPass(createConvertSCFToCFPass());
  
  // Lower to LLVM
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  
  if (failed(pm.run(module)))
    return llvm::make_error<llvm::StringError>("Optimization pipeline failed", llvm::inconvertibleErrorCode());

  // 2. Translate MLIR to LLVM IR
  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  auto llvmModule = translateModuleToLLVMIR(module, *llvmContext);
  if (!llvmModule)
    return llvm::make_error<llvm::StringError>("Failed to translate MLIR to LLVM IR", llvm::inconvertibleErrorCode());

  // 3. Add to JIT
  if (auto err = jit->addIRModule(llvm::orc::ThreadSafeModule(std::move(llvmModule), std::move(llvmContext))))
    return err;

  // 4. Execute 'main'
  return invoke("main").takeError();
}

llvm::Expected<int> JitRunner::invoke(llvm::StringRef functionName) {
  auto sym = jit->lookup(functionName);
  if (!sym)
    return sym.takeError();

  auto fn = (int (*)())(intptr_t)sym->getValue();
  return fn();
}

llvm::Expected<void*> JitRunner::lookup(llvm::StringRef name) {
  auto sym = jit->lookup(name);
  if (!sym)
    return sym.takeError();
  return (void*)(intptr_t)sym->getValue();
}

llvm::Error JitRunner::compile(ModuleOp module) {
  // 1. Run optimization pipeline
  PassManager pm(module.getContext());
  
  // Lower TensorLang to Linalg
  pm.addPass(createConvertTensorLangToLinalgPass());

  // Bufferize
  bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  pm.addPass(bufferization::createOneShotBufferizePass(options));
  pm.addPass(createCanonicalizerPass());

  // Lower Linalg -> SCF loops
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(memref::createExpandStridedMetadataPass());

  // Lower SCF -> ControlFlow
  pm.addPass(createConvertSCFToCFPass());

  // Lower to LLVM
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  if (failed(pm.run(module)))
    return llvm::make_error<llvm::StringError>("Optimization pipeline failed", llvm::inconvertibleErrorCode());

  // 2. Translate MLIR to LLVM IR
  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  auto llvmModule = translateModuleToLLVMIR(module, *llvmContext);
  if (!llvmModule)
    return llvm::make_error<llvm::StringError>("Failed to translate MLIR to LLVM IR", llvm::inconvertibleErrorCode());

  // 3. Add to JIT
  auto& ES = jit->getExecutionSession();
  auto dylibOrErr = ES.createJITDylib("fix_" + std::to_string(dylibCount++));
  if (!dylibOrErr) return dylibOrErr.takeError();
  auto& dylib = *dylibOrErr;

  dylib.addGenerator(
      cantFail(llvm::orc::EPCDynamicLibrarySearchGenerator::GetForTargetProcess(ES)));
  
  // Make sure new dylib can see symbols in main dylib
  dylib.addToLinkOrder(jit->getMainJITDylib());
  
  // Update the search order of main dylib to look into the new dylib FIRST
  // This is how we achieve hot-swapping
  jit->getMainJITDylib().addToLinkOrder(dylib, llvm::orc::JITDylibLookupFlags::MatchAllSymbols);

  return jit->addIRModule(dylib, llvm::orc::ThreadSafeModule(std::move(llvmModule), std::move(llvmContext)));
}

#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

// ...

llvm::Error JitRunner::compileString(llvm::StringRef source) {
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  
  // Register bufferization interfaces
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);
  
  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<TensorLangDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<tensor::TensorDialect>();
  context.getOrLoadDialect<linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::vector::VectorDialect>();
  context.getOrLoadDialect<mlir::affine::AffineDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<LLVM::LLVMDialect>();
  
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(source), llvm::SMLoc());
  
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module)
    return llvm::make_error<llvm::StringError>("Failed to parse MLIR source", llvm::inconvertibleErrorCode());

  return compile(*module);
}
