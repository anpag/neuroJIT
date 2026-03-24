#include "tensorlang/Runtime/ASTMutator.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/Verifier.h"
#include <regex>
#include <iostream>

namespace mlir {
namespace tensorlang {

std::optional<mlir::ModuleOp> ASTMutator::mutateConstant(mlir::ModuleOp baseModule, llvm::StringRef funcName, int constantIndex, float newValue) {
  mlir::ModuleOp clonedModule = baseModule.clone();
  
  auto funcOp = clonedModule.lookupSymbol<mlir::func::FuncOp>(funcName);
  if (!funcOp) {
    clonedModule.erase();
    return std::nullopt;
  }

  int currentIndex = 0;
  mlir::arith::ConstantOp targetOp;
  
  funcOp.walk([&](mlir::arith::ConstantOp op) {
    // Only count float constants for simplicity, based on newValue being float
    if (mlir::isa<mlir::FloatType>(op.getType())) {
      if (currentIndex == constantIndex) {
        targetOp = op;
      }
      currentIndex++;
    }
  });

  if (!targetOp) {
    clonedModule.erase();
    return std::nullopt;
  }

  mlir::OpBuilder builder(targetOp);
  auto newConst = builder.create<mlir::arith::ConstantOp>(
      targetOp.getLoc(), targetOp.getType(), builder.getF32FloatAttr(newValue));
  
  targetOp.replaceAllUsesWith(newConst.getResult());
  targetOp.erase();

  return clonedModule;
}

std::optional<mlir::ModuleOp> ASTMutator::swapBinaryOperator(mlir::ModuleOp baseModule, llvm::StringRef funcName, int opIndex, llvm::StringRef newOpType) {
  mlir::ModuleOp clonedModule = baseModule.clone();
  
  auto funcOp = clonedModule.lookupSymbol<mlir::func::FuncOp>(funcName);
  if (!funcOp) {
    clonedModule.erase();
    return std::nullopt;
  }

  int currentIndex = 0;
  mlir::Operation* targetOp = nullptr;
  
  funcOp.walk([&](mlir::Operation* op) {
    // Only target ops that have 2 operands and 1 result (likely binary ops)
    if (op->getNumOperands() == 2 && op->getNumResults() == 1 &&
        op->getName().getStringRef().starts_with("arith.") &&
        !llvm::isa<mlir::arith::CmpFOp, mlir::arith::CmpIOp>(op)) {
      if (currentIndex == opIndex) {
        targetOp = op;
      }
      currentIndex++;
    }
  });

  if (!targetOp) {
    clonedModule.erase();
    return std::nullopt;
  }

  mlir::OpBuilder builder(targetOp);
  mlir::OperationState state(targetOp->getLoc(), newOpType);
  state.addOperands(targetOp->getOperands());
  state.addTypes(targetOp->getResultTypes());
  
  // Ensure the new operation name is registered/valid
  mlir::OperationName opName(newOpType, clonedModule.getContext());
  if (!opName.isRegistered()) {
    clonedModule.erase();
    return std::nullopt;
  }

  mlir::Operation* newOp = builder.create(state);
  
  targetOp->replaceAllUsesWith(newOp);
  targetOp->erase();

  return clonedModule;
}

std::optional<mlir::ModuleOp> ASTMutator::applyMutations(mlir::ModuleOp baseModule, llvm::StringRef jsonCommand) {
  std::string jsonStr = jsonCommand.str();
  
  std::smatch match;
  std::string targetFuncName = "matmul"; // Default
  std::regex target_regex(R"regex("target_function"\s*:\s*"([^"]+)")regex");
  if (std::regex_search(jsonStr, match, target_regex)) {
    targetFuncName = match[1].str();
  }

  mlir::ModuleOp clonedModule = baseModule.clone();
  auto funcOp = clonedModule.lookupSymbol<mlir::func::FuncOp>(targetFuncName);
  
  if (!funcOp) {
    std::cerr << "[ASTMutator] Target function '" << targetFuncName << "' not found.\n";
    clonedModule.erase();
    return std::nullopt;
  }

  // Parse unroll
  std::regex unroll_regex(R"regex("type"\s*:\s*"unroll".*?"factor"\s*:\s*(\d+))regex");
  if (std::regex_search(jsonStr, match, unroll_regex)) {
    uint64_t unrollFactor = std::stoull(match[1].str());
    
    mlir::affine::AffineForOp targetLoop;
    funcOp.walk([&](mlir::affine::AffineForOp forOp) {
      if (!targetLoop) targetLoop = forOp; // take outermost
    });
    
    if (targetLoop && unrollFactor > 1) {
      if (mlir::failed(mlir::affine::loopUnrollByFactor(targetLoop, unrollFactor))) {
        std::cerr << "[ASTMutator] loopUnrollByFactor failed.\n";
      } else {
        std::cout << "[ASTMutator] Applied unrollLoop with factor " << unrollFactor << "\n";
      }
    }
  }

  // Parse tile
  std::regex tile_regex(R"regex("type"\s*:\s*"tile".*?"sizes"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\])regex");
  if (std::regex_search(jsonStr, match, tile_regex)) {
    llvm::SmallVector<unsigned, 3> tileSizes;
    tileSizes.push_back(std::stoul(match[1].str()));
    tileSizes.push_back(std::stoul(match[2].str()));
    tileSizes.push_back(std::stoul(match[3].str()));

    llvm::SmallVector<mlir::affine::AffineForOp, 3> band;
    funcOp.walk([&](mlir::affine::AffineForOp forOp) {
      band.push_back(forOp);
    });

    if (!band.empty()) {
      llvm::SmallVector<mlir::affine::AffineForOp, 3> tiledBand;
      if (mlir::failed(mlir::affine::tilePerfectlyNested(band, tileSizes, &tiledBand))) {
        std::cerr << "[ASTMutator] tilePerfectlyNested failed.\n";
      } else {
        std::cout << "[ASTMutator] Applied tileLoop with sizes [" << tileSizes[0] << ", " << tileSizes[1] << ", " << tileSizes[2] << "]\n";
      }
    }
  }

  if (mlir::failed(mlir::verify(clonedModule))) {
    std::cerr << "[ASTMutator] Verification failed after mutations.\n";
    clonedModule.erase();
    return std::nullopt;
  }

  return clonedModule;
}

} // namespace tensorlang
} // namespace mlir
