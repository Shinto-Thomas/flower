"""fltabular: ClientApp with client-side fixed clipping for central DP."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.clientapp.mod import fixedclipping_mod

from fltabular.task import GLM, load_data, test_fn, train_fn

app = ClientApp()


@app.train(mods=[fixedclipping_mod])
def train(msg: Message, context: Context) -> Message:
    """Local training. fixedclipping_mod clips this client's update to the
    server-broadcast clipping norm before it's sent back — required for the
    server's DifferentialPrivacyClientSideFixedClipping wrapper."""

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    csv_path = context.run_config["dataset-path"]
    group_col = context.run_config["group-col"]
    target_col = context.run_config["target-col"]
    batch_size = int(context.run_config["batch-size"])

    trainloader, _, _, num_features = load_data(
        csv_path, group_col, target_col, partition_id, num_partitions, batch_size
    )

    net = GLM(num_features)
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    lr = float(msg.content["config"]["lr"])
    local_epochs = int(context.run_config["local-epochs"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loss = train_fn(net, trainloader, local_epochs, lr, device)

    content = RecordDict(
        {
            "arrays": ArrayRecord(net.state_dict()),
            "metrics": MetricRecord(
                {"train_loss": train_loss, "num-examples": len(trainloader.dataset)}
            ),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate on local val split each round; on the final round, switch to
    the held-out test split for the true, only-seen-once test metric."""

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    csv_path = context.run_config["dataset-path"]
    group_col = context.run_config["group-col"]
    target_col = context.run_config["target-col"]
    batch_size = int(context.run_config["batch-size"])
    num_server_rounds = int(context.run_config["num-server-rounds"])

    _, valloader, testloader, num_features = load_data(
        csv_path, group_col, target_col, partition_id, num_partitions, batch_size
    )

    net = GLM(num_features)
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    server_round = int(msg.content["config"]["server-round"])
    is_final_round = server_round == num_server_rounds
    loader = testloader if is_final_round else valloader
    tag = "test" if is_final_round else "val"

    loss, metrics = test_fn(net, loader, device)

    content = RecordDict(
        {
            "metrics": MetricRecord(
                {
                    f"{tag}_loss": loss,
                    f"{tag}_mae": metrics["mae"],
                    f"{tag}_rmse": metrics["rmse"],
                    f"{tag}_r2": metrics["r2"],
                    "num-examples": metrics["num_examples"],
                }
            )
        }
    )
    return Message(content=content, reply_to=msg)