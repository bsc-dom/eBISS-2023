from model.text import TextFile
from dataclay import Client
from dataclay.exceptions import DataClayException

client = Client(host="127.0.0.1", username="testuser", password="s3cret", dataset="testdata")
client.start()

try:
    TextFile.delete_alias("Romeo and Juliet")
except DataClayException:
    pass

try:
    TextFile.delete_alias("The Tempest")
except DataClayException:
    pass

try:
    TextFile.delete_alias("A Midsummer Night's Dream")
except DataClayException:
    pass
