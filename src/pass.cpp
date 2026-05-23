#include "core.h"
#include "gpuSpec.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

namespace {

struct MyPass : public PassWrapper<MyPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyPass)

    MyPass() = default;
    MyPass(const MyPass &other) : PassWrapper(other) {}

    StringRef getArgument() const final { return "my-cost-analysis"; }
    StringRef getDescription() const final {
        return "Build and print a symbolic cost expression for func.func @main";
    }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<arith::ArithDialect, func::FuncDialect>();
    }

    // CLI entry function funcName argument, default to main
    Option<std::string> funcName{
        *this,
            "func-name",
            llvm::cl::desc("Function/symbol name to analyze"),
            llvm::cl::init("main")
    };

    void runOnOperation() final {
        triton::FuncOp mainFunc;
        getOperation().walk([&](triton::FuncOp funcOp) {
            if (funcOp.getSymName() == funcName) {
                mainFunc = funcOp;
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });

        if (!mainFunc) {
            llvm::errs() << "No func.func @main found\n";
            return;
        }

        analyze_cost(*mainFunc, llvm::errs());
    }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
    return {MLIR_PLUGIN_API_VERSION, "My Pass", "v0.1", []() {
                PassRegistration<MyPass>();
            }};
}
