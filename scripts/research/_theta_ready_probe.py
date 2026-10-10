from utils.api.thetadata.thetadata_client import get_thetadata_client

c = get_thetadata_client()
print("ready", c.is_ready())
