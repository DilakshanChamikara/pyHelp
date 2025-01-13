import hashlib
import random
import string
import time
from argon2 import PasswordHasher


# Function to generate a random string of a given length
def generate_random_string(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


# Function to compute 1000 hashes with a given hash function
def compute_hashes(hash_function, count=1000):
    random_strings = [generate_random_string(16) for _ in range(count)]
    start_time = time.time()
    hashes = [hash_function(value.encode()) for value in random_strings]
    end_time = time.time()
    return hashes, end_time - start_time


# Hash functions
def md5_hash(data):
    return hashlib.md5(data).hexdigest()


def sha1_hash(data):
    return hashlib.sha1(data).hexdigest()


def sha256_hash(data):
    return hashlib.sha256(data).hexdigest()


# Argon2id hashing
def argon2id_hash(data):
    ph = PasswordHasher()
    return ph.hash(data.decode())


# Main function to compute and compare hash timings
if __name__ == "__main__":
    count = 1000  # Number of hashes to compute

    print("Computing 1000 hashes for each algorithm...\n")

    # MD5
    _, md5_time = compute_hashes(md5_hash, count)
    print(f"MD5 Time: {md5_time:.2f} seconds")

    # SHA-1
    _, sha1_time = compute_hashes(sha1_hash, count)
    print(f"SHA-1 Time: {sha1_time:.2f} seconds")

    # SHA-256
    _, sha256_time = compute_hashes(sha256_hash, count)
    print(f"SHA-256 Time: {sha256_time:.2f} seconds")

    # Argon2id
    try:
        _, argon2id_time = compute_hashes(argon2id_hash, count)
        print(f"Argon2id Time: {argon2id_time:.2f} seconds")
    except Exception as e:
        print(f"Error with Argon2id: {e}")
