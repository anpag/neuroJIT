#include "tensorlang/Conversion/TensorLangToLinalg/TensorLangToLinalg.h"

#include "tensorlang/Dialect/TensorLang/IR/TensorLangOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tensorlang;

namespace {

class TensorLangTypeConverter : public TypeConverter {
public:
  TensorLangTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](LinearTensorType type) -> Type {
      return RankedTensorType::get(type.getShape(), type.getElementType());
    });
  }
};

//===----------------------------------------------------------------------===//
// MatMulLowering
//===----------------------------------------------------------------------===//

struct MatMulLowering : public OpConversionPattern<MatMulOp> {
  using OpConversionPattern<MatMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MatMulOp op, OpAdaptor adaptor, 
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    
    Type convertedResultType = getTypeConverter()->convertType(op.getResult().getType());
    auto tensorType = dyn_cast<RankedTensorType>(convertedResultType);
    if (!tensorType)
      return rewriter.notifyMatchFailure(op, "result must be a ranked tensor");

    // 1. Create an empty tensor for the output (init operand for linalg)
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
    Value initTensor = rewriter.create<tensor::EmptyOp>(loc, tensorType.getShape(), tensorType.getElementType());
    Value filledTensor = rewriter.create<linalg::FillOp>(loc, zero, initTensor).getResult(0);

    // 2. Create the linalg.matmul op
    auto matmulOp = rewriter.create<linalg::MatmulOp>(
        loc,
        tensorType,
        ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        ValueRange{filledTensor});

    rewriter.replaceOp(op, matmulOp.getResult(0));
    return success();
  }
};

struct ConstantOpLowering : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // tensorlang.constant -> arith.constant
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValue());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct ConvertTensorLangToLinalgPass
    : public PassWrapper<ConvertTensorLangToLinalgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTensorLangToLinalgPass)

  StringRef getArgument() const override { return "convert-tensorlang-to-linalg"; }
  StringRef getDescription() const override { return "Lower TensorLang ops to Linalg"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    TensorLangTypeConverter typeConverter;
    
    ConversionTarget target(*context);
    
    // We want to convert to Linalg
    target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect, arith::ArithDialect>();
    
    // FuncOp is legal only if signature is legal
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });

    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return typeConverter.isLegal(op);
    });

    // TensorLang ops are illegal
    target.addIllegalOp<MatMulOp>();
    target.addIllegalOp<ConstantOp>();

    RewritePatternSet patterns(context);
    patterns.add<MatMulLowering, ConstantOpLowering>(typeConverter, context);
    
    // Add function signature conversion patterns
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::tensorlang::createConvertTensorLangToLinalgPass() {
  return std::make_unique<ConvertTensorLangToLinalgPass>();
}

void mlir::tensorlang::registerConvertTensorLangToLinalgPass() {
  PassRegistration<ConvertTensorLangToLinalgPass>();
}
