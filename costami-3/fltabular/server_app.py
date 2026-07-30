"""fltabular: ServerApp — FedAvg wrapped with central DP (client-side fixed clipping)."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import DifferentialPrivacyClientSideFixedClipping, FedAvg

from fltabular.task import GLM

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # --- run config -------------------------------------------------------
    num_rounds = int(context.run_config["num-server-rounds"])
    fraction_train = float(context.run_config["fraction-train"])
    fraction_evaluate = float(context.run_config["fraction-evaluate"])
    lr = float(context.run_config["learning-rate"])

    # --- DP config ----------------------------------------------------------
    noise_multiplier = float(context.run_config["noise-multiplier"])
    clipping_norm = float(context.run_config["clipping-norm"])
    num_sampled_clients = int(context.run_config["num-sampled-clients"])

    # --- model / features ---------------------------------------------------
    # num-features must equal len(feature_cols) computed in task.py, i.e.
    # (total columns) - group_col - target_col.
    num_features = int(context.run_config["num-features"])
    global_model = GLM(num_features)
    arrays = ArrayRecord(global_model.state_dict())

    # --- base strategy, wrapped with central DP fixed-clipping --------------
    base_strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )
    dp_strategy = DifferentialPrivacyClientSideFixedClipping(
        base_strategy,
        noise_multiplier,
        clipping_norm,
        num_sampled_clients,
    )

    # --- run federated training ---------------------------------------------
    result = dp_strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # --- persist final (DP-noised) global model -----------------------------
    if context.run_config.get("save-model", True):
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")