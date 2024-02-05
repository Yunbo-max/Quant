from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage, Feed, Market
from typing import List

client = WebSocketClient(
    api_key="<ocunxnOqC0pnltRqT3VkOiKeCmPE49L7>", feed=Feed.Launchpad, market=Market.Stocks
)

client.subscribe("AM.*")  # all aggregates
# client.subscribe("LV.*")  # all aggregates
# client.subscribe("AM.O:A230616C00070000")  # all aggregates
# client.subscribe("LV.O:A230616C00070000")  # all aggregates


def handle_msg(msgs: List[WebSocketMessage]):
    for m in msgs:
        print(m)


# print messages
client.run(handle_msg)
