"""fltabular: Flower Example on Adult Census Income Tabular Dataset."""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fltabular.task import CostRegressor, evaluator, load_data, trainer


app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train on local data."""

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    train_loader, _ = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
    )

    net = CostRegressor()
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    trainer(net, train_loader)

    model_record = ArrayRecord(net.state_dict())

    metrics = {
        "num-examples": len(train_loader),
    }

    metric_record = MetricRecord(metrics)

    content = RecordDict({
        "arrays": model_record,
        "metrics": metric_record,
    })

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate on local data."""

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    _, test_loader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
    )

    net = CostRegressor()
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    mae, mse, rmse, r2 = evaluator(net, test_loader)

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "num-examples": len(test_loader),
    }

    metric_record = MetricRecord(metrics)

    content = RecordDict({
        "metrics": metric_record
    })

    return Message(content=content, reply_to=msg)