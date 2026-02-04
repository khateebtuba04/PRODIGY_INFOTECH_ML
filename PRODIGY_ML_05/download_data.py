import requests
import zipfile
import os
import io

def download_and_extract(url, target_dir):
    print(f"Downloading from {url}...")
    try:
        r = requests.get(url)
        r.raise_for_status()
        print("Download complete. Extracting...")
        
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(target_dir)
        print(f"Extracted to {target_dir}")
        
        # List extracted directories to confirm
        print("Contents of data directory:")
        for root, dirs, files in os.walk(target_dir):
            for d in dirs:
                print(os.path.join(root, d))
            break
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    url = "https://github.com/srohit0/food_mnist/archive/refs/heads/master.zip"
    target_dir = "data"
    os.makedirs(target_dir, exist_ok=True)
    download_and_extract(url, target_dir)
