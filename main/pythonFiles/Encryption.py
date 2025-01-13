def caesar_cipher(text, shift):
    result = ""
    for char in text:
        if char.isalpha():  # Check if the character is a letter
            shift_base = ord('A') if char.isupper() else ord('a')
            # Shift character and wrap around using modulo
            result += chr((ord(char) - shift_base + shift) % 26 + shift_base)
        else:
            result += char  # Leave non-alphabetic characters unchanged
    return result

# Example usage
text = str(input("Enter the text to shift: "))
shift = int(input("Enter the shift key: "))
shifted_text = caesar_cipher(text, shift)
print("Shifted text:", shifted_text)