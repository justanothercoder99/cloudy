from env import BUCKET_NAME
from google.cloud import storage
import io

class StorageHandler:
    def __init__(self) -> None:
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(BUCKET_NAME)
        self.init = True

    def uploadToCloud(self, df, table_name):
        if self.init != True:
            print("Storage Handler not initialised")
        blob = self.bucket.blob(table_name)

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, header=False)

        # Upload the CSV contents
        blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
    
    def getBlobs(self):
        return self.storage_client.list_blobs(self.bucket.name)