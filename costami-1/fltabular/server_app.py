"""fltabular: Flower Example on Adult Census Income Tabular Dataset."""

from time import perf_counter

import torch
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
import flwr.serverapp.strategy.result as strategy_result

from fltabular.task import CostRegressor, get_input_dim


def _format_value_fixed(val):
    if isinstance(val, float):
        return f"{val:.4f}"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, list):
        return str([
            f"{x:.4f}" if isinstance(x, float) else str(x)
            for x in val
        ])
    return str(val)

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    strategy_result.format_value = _format_value_fixed

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]

    # Init global model
    net = CostRegressor(get_input_dim())
    arrays = ArrayRecord(net.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg()

    # Start strategy, run FedAvg for `num_rounds`
    start_time = perf_counter()
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )
    elapsed_seconds = perf_counter() - start_time
    print(f"Federated run time: {elapsed_seconds:.2f} seconds")

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
