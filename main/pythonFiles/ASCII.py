# Function to convert decimal (ASCII) to binary
def decimal_to_binary(decimal_number):
    return bin(decimal_number)[2:]  # bin() converts to binary with a '0b' prefix

# Function to convert binary to decimal (ASCII)
def binary_to_decimal(binary_str):
    return int(binary_str, 2)  # Convert binary string to decimal

# Function to convert sentence to ASCII values and then to binary
def sentence_to_ascii_and_binary(sentence):
    ascii_values = [ord(char) for char in sentence]  # Convert each character to ASCII
    binary_values = [decimal_to_binary(ascii_val).zfill(8) for ascii_val in ascii_values]  # Convert each ASCII to 8-bit binary
    return ascii_values, binary_values

# Function to convert binary back to sentence
def binary_to_sentence(binary_values):
    # Split the continuous binary string into chunks of 8 bits (1 byte per character)
    chars = [chr(binary_to_decimal(binary_values[i:i+8])) for i in range(0, len(binary_values), 8)]
    return ''.join(chars)

# Function to perform XOR on two binary strings
def xor_binary(bin1, bin2):
    # Ensure both binaries are the same length by padding the shorter one with leading zeros
    max_length = max(len(bin1), len(bin2))
    bin1 = bin1.zfill(max_length)
    bin2 = bin2.zfill(max_length)

    # Perform XOR on each bit and store the result
    xor_result = ''.join('1' if bin1[i] != bin2[i] else '0' for i in range(max_length))

    return xor_result

# Example usage
sentence = input("Enter a sentence: ")

# Convert the sentence to ASCII and binary
ascii_values, binary_values = sentence_to_ascii_and_binary(sentence)

# Convert the binary values back to a sentence
binary_string = ''.join(binary_values)  # Concatenate the binary values into one string
reconstructed_sentence = binary_to_sentence(binary_string)

# Display the results
ascii_result = ''.join(str(val) for val in ascii_values)
binary_result = ''.join(str(val) for val in binary_values)

# Passing data to XOR function
binary1 = binary_result
sentence2 = str(input("Enter Letters for XOR: "))
ascii_values2, binary_values2 = sentence_to_ascii_and_binary(sentence2)
binary_result2 = ''.join(str(val) for val in binary_values2)
binary2 = ''.join(binary_result2)

getLength = len(binary_result2)

reconstructed_sentence2 = binary_to_sentence(binary2)

# Perform XOR on the two binary strings
xor_result = xor_binary(binary1, binary2)

print(f"ASCII values: {ascii_result}")
print(f"Binary values 1: {binary_result}")
print(f"Binary values 2: {binary_result2}")
print(f"Length of Binary value 2: {getLength}")
print(f"XOR result: {xor_result}")
print(f"Reconstructed Sentence 1: {reconstructed_sentence}")
print(f"Reconstructed Sentence 2: {reconstructed_sentence2}")
