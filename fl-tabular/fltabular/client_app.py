"""fltabular: Flower Example on Adult Census Income Tabular Dataset."""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fltabular.task import CostRegressor, evaluator, get_input_dim, load_data, trainer

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Load dataset
    partition_id = context.node_config["partition-id"]
    train_loader = load_data(
        "train",
        partition_id=partition_id, num_partitions=context.node_config["num-partitions"]
    )
    print(f"Client {partition_id} train samples: {len(train_loader.dataset)}")

    # Load model
    net = CostRegressor(get_input_dim())
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Perform training
    trainer(net, train_loader)
    train_mae, train_mse, train_rmse, train_r2 = evaluator(net, train_loader)

    # Construct and return reply Message
    model_record = ArrayRecord(net.state_dict())
    metrics = {
        "num-examples": len(train_loader.dataset),
        "train-mae": train_mae,
        "train-mse": train_mse,
        "train-rmse": train_rmse,
        "train-r2": train_r2,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    # Load dataset
    partition_id = context.node_config["partition-id"]
    test_loader = load_data(
        "val",
        partition_id=partition_id, num_partitions=context.node_config["num-partitions"]
    )
    print(f"Client {partition_id} val samples (global test set): {len(test_loader.dataset)}")

    # Load model
    net = CostRegressor(get_input_dim())
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Perform evaluation
    mae, mse, rmse, r2 = evaluator(net, test_loader)

    # Construct and return reply Message
    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "num-examples": len(test_loader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
