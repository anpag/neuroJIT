#include "tensorlang/ExecutionEngine/JitRunner.h"
#include "tensorlang/Conversion/TensorLangToLinalg/TensorLangToLinalg.h"
#include "tensorlang/Dialect/TensorLang/Transforms/Passes.h"
#include "tensorlang/Dialect/TensorLang/IR/TensorLangDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
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
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"

// Correct Header for EPCDynamicLibrarySearchGenerator
#include "llvm/ExecutionEngine/Orc/EPCDynamicLibrarySearchGenerator.h"

using namespace mlir;
using namespace mlir::tensorlang;

static std::string g_last_error_msg;

extern "C" {
const char* tensorlang_get_last_jit_error() {
  return g_last_error_msg.c_str();
}
}

JitRunner::JitRunner(std::unique_ptr<llvm::orc::LLJIT> jit) : jit(std::move(jit)) {}
JitRunner::~JitRunner() = default;

llvm::Expected<std::unique_ptr<JitRunner>> JitRunner::create() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto jitBuilder = llvm::orc::LLJITBuilder();
  auto jitOrError = jitBuilder.create();
  if (auto error = jitOrError.takeError()) return std::move(error);
  auto jit = std::move(*jitOrError);
  jit->getMainJITDylib().addGenerator(
      cantFail(llvm::orc::EPCDynamicLibrarySearchGenerator::GetForTargetProcess(
          jit->getExecutionSession())));
  return std::unique_ptr<JitRunner>(new JitRunner(std::move(jit)));
}

llvm::Error JitRunner::run(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createVerifyLinearity());
  pm.addPass(createConvertTensorLangToLinalgPass());
  bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  pm.addPass(bufferization::createOneShotBufferizePass(options));
  pm.addPass(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  if (failed(pm.run(module))) return llvm::make_error<llvm::StringError>("Optimization failed", llvm::inconvertibleErrorCode());
  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  auto llvmModule = translateModuleToLLVMIR(module, *llvmContext);
  if (!llvmModule) return llvm::make_error<llvm::StringError>("Translation failed", llvm::inconvertibleErrorCode());
  if (auto err = jit->addIRModule(llvm::orc::ThreadSafeModule(std::move(llvmModule), std::move(llvmContext)))) return err;
  return invoke("main").takeError();
}

llvm::Expected<int> JitRunner::invoke(llvm::StringRef functionName) {
  auto sym = jit->lookup(functionName);
  if (!sym) return sym.takeError();
  auto fn = (int (*)())(intptr_t)sym->getValue();
  return fn();
}

llvm::Expected<void*> JitRunner::lookup(llvm::StringRef name) {
  auto sym = jit->lookup(name);
  if (!sym) return sym.takeError();
  return (void*)(intptr_t)sym->getValue();
}

llvm::Error JitRunner::compile(ModuleOp module) {
  g_last_error_msg = "";
  std::string diag_str;
  llvm::raw_string_ostream os(diag_str);
  auto handler = module->getContext()->getDiagEngine().registerHandler([&](mlir::Diagnostic &diag) {
    os << diag << "\n";
    return mlir::success();
  });

  PassManager pm(module.getContext());
  pm.addPass(createConvertTensorLangToLinalgPass());
  bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  pm.addPass(bufferization::createOneShotBufferizePass(options));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  if (failed(pm.run(module))) {
    g_last_error_msg = diag_str;
    module->getContext()->getDiagEngine().eraseHandler(handler);
    return llvm::make_error<llvm::StringError>("Optimization failed: " + diag_str, llvm::inconvertibleErrorCode());
  }

  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  auto llvmModule = translateModuleToLLVMIR(module, *llvmContext);
  if (!llvmModule) {
    g_last_error_msg = "Translation failed";
    module->getContext()->getDiagEngine().eraseHandler(handler);
    return llvm::make_error<llvm::StringError>(g_last_error_msg, llvm::inconvertibleErrorCode());
  }

  auto& ES = jit->getExecutionSession();
  auto dylibOrErr = ES.createJITDylib("fix_" + std::to_string(dylibCount++));
  if (!dylibOrErr) { module->getContext()->getDiagEngine().eraseHandler(handler); return dylibOrErr.takeError(); }
  auto& dylib = *dylibOrErr;
  dylib.addGenerator(cantFail(llvm::orc::EPCDynamicLibrarySearchGenerator::GetForTargetProcess(ES)));
  auto& mainDylib = jit->getMainJITDylib();
  mainDylib.withLinkOrderDo([&](const llvm::orc::JITDylibSearchOrder& currentOrder) {
      llvm::orc::JITDylibSearchOrder newOrder;
      newOrder.push_back({&dylib, llvm::orc::JITDylibLookupFlags::MatchAllSymbols});
      for (auto& entry : currentOrder) { if (entry.first != &dylib) newOrder.push_back(entry); }
      mainDylib.setLinkOrder(newOrder);
  });
  auto err = jit->addIRModule(dylib, llvm::orc::ThreadSafeModule(std::move(llvmModule), std::move(llvmContext)));
  module->getContext()->getDiagEngine().eraseHandler(handler);
  return err;
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

llvm::Error JitRunner::compileString(llvm::StringRef source) {
  g_last_error_msg = "";
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
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
  if (!module) {
    g_last_error_msg = "Failed to parse MLIR source";
    return llvm::make_error<llvm::StringError>(g_last_error_msg, llvm::inconvertibleErrorCode());
  }
  return compile(*module);
}
