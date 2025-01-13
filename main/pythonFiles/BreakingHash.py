import hashlib

# sha1
def crack_sha1_hash(hash_to_crack, pin_length):
    # Iterate through all possible PIN combinations
    for pin in range(10 ** pin_length):
        # Format the PIN with leading zeros (e.g., 0001 for a 4-digit PIN)
        pin_str = str(pin).zfill(pin_length)

        # Hash the PIN and compare with the target hash
        hashed_pin = hashlib.sha1(pin_str.encode()).hexdigest()
        if hashed_pin == hash_to_crack:
            return pin_str  # Return the cracked PIN

    return None  # If no match is found

# md5
def crack_md5_hash(hash_to_crack, pin_length):
    # Iterate through all possible PIN combinations
    for pin in range(10 ** pin_length):
        # Format the PIN with leading zeros (e.g., 0001 for a 4-digit PIN)
        pin_str = str(pin).zfill(pin_length)

        # Hash the PIN using MD5 and compare with the target hash
        hashed_pin = hashlib.md5(pin_str.encode()).hexdigest()
        if hashed_pin == hash_to_crack:
            return pin_str  # Return the cracked PIN

    return None  # If no match is found


# Example usage
sha1_hash_to_crack = "30139264c3ec85759ce4f83c2fe286ecb63e6d43"
pin1_length = 4  # Length of the PIN code

md5_hash_to_crack = "c49078e81caafab96c08390197cf6a96"
pin2_length = 4  # Length of the PIN code

# Crack the hash
cracked_pin1 = crack_sha1_hash(sha1_hash_to_crack, pin1_length)
cracked_pin2 = crack_md5_hash(md5_hash_to_crack, pin2_length)

# Display the result
if cracked_pin1:
    print(f"The PIN code is: {cracked_pin1}")
else:
    print("Failed to crack the hash.")

if cracked_pin2:
    print(f"The PIN code is: {cracked_pin2}")
else:
    print("Failed to crack the hash.")