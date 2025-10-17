"""
Phase 1: Baseline ECC + LSB Steganography
Implements basic public-key steganography with ECC keys and LSB embedding
"""

import os
import time
import hashlib
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from PIL import Image
import numpy as np
import psutil


class ECCKeyManager:
    """Manages ECC key generation and serialization"""
    
    @staticmethod
    def generate_keypair(save_path=None):
        """Generate ECC keypair using Curve25519"""
        private_key = ec.generate_private_key(ec.SECP256R1())
        public_key = private_key.public_key()
        
        if save_path:
            # Save private key
            with open(f"{save_path}_private.pem", "wb") as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Save public key
            with open(f"{save_path}_public.pem", "wb") as f:
                f.write(public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
        
        return private_key, public_key
    
    @staticmethod
    def load_private_key(path):
        """Load private key from PEM file"""
        with open(path, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)
    
    @staticmethod
    def load_public_key(path):
        """Load public key from PEM file"""
        with open(path, "rb") as f:
            return serialization.load_pem_public_key(f.read())


class ECCCrypto:
    """Handles ECC-based encryption/decryption"""
    
    @staticmethod
    def derive_shared_key(private_key, peer_public_key):
        """Perform ECDH and derive AES key using HKDF"""
        shared_secret = private_key.exchange(ec.ECDH(), peer_public_key)
        
        # Derive 256-bit AES key using HKDF
        derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'ecc-stego-key'
        ).derive(shared_secret)
        
        return derived_key
    
    @staticmethod
    def encrypt_message(message, aes_key):
        """Encrypt message using AES-256-GCM"""
        if isinstance(message, str):
            message = message.encode('utf-8')
        
        # Generate random IV (12 bytes for GCM)
        iv = os.urandom(12)
        
        # Create cipher
        cipher = Cipher(algorithms.AES(aes_key), modes.GCM(iv))
        encryptor = cipher.encryptor()
        
        # Encrypt
        ciphertext = encryptor.update(message) + encryptor.finalize()
        
        # Return IV + tag + ciphertext
        return iv + encryptor.tag + ciphertext
    
    @staticmethod
    def decrypt_message(encrypted_data, aes_key):
        """Decrypt AES-GCM encrypted message"""
        # Extract IV (12 bytes), tag (16 bytes), and ciphertext
        iv = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]
        
        # Create cipher
        cipher = Cipher(algorithms.AES(aes_key), modes.GCM(iv, tag))
        decryptor = cipher.decryptor()
        
        # Decrypt
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        return plaintext.decode('utf-8')


class LSBSteganography:
    """LSB-based image steganography"""
    
    @staticmethod
    def embed_data(cover_image_path, data, output_path):
        """Embed data into image using LSB"""
        img = Image.open(cover_image_path).convert('RGB')
        img_array = np.array(img)
        
        # Convert data to binary string
        data_len = len(data)
        binary_data = ''.join(format(byte, '08b') for byte in data)
        
        # Add length header (32 bits)
        length_header = format(data_len, '032b')
        binary_data = length_header + binary_data
        
        # Check capacity
        max_capacity = img_array.size
        if len(binary_data) > max_capacity:
            raise ValueError(f"Data too large: {len(binary_data)} bits > {max_capacity} bits")
        
        # Flatten and embed
        flat_img = img_array.flatten()
        for i, bit in enumerate(binary_data):
            flat_img[i] = (flat_img[i] & 0xFE) | int(bit)
        
        # Reshape and save
        stego_img = flat_img.reshape(img_array.shape)
        Image.fromarray(stego_img.astype('uint8')).save(output_path, 'PNG')
        
        return output_path
    
    @staticmethod
    def extract_data(stego_image_path):
        """Extract data from stego image"""
        img = Image.open(stego_image_path).convert('RGB')
        img_array = np.array(img).flatten()
        
        # Extract length (first 32 bits)
        length_bits = ''.join(str(pixel & 1) for pixel in img_array[:32])
        data_len = int(length_bits, 2)
        
        # Extract data
        start_idx = 32
        end_idx = start_idx + (data_len * 8)
        data_bits = ''.join(str(pixel & 1) for pixel in img_array[start_idx:end_idx])
        
        # Convert to bytes
        data = bytearray()
        for i in range(0, len(data_bits), 8):
            byte = data_bits[i:i+8]
            data.append(int(byte, 2))
        
        return bytes(data)


class BaselineBenchmark:
    """Benchmarking utilities"""
    
    @staticmethod
    def measure_memory():
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def hash_data(data):
        """Generate SHA256 hash"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()


def main():
    """Main demonstration of Phase 1"""
    print("=" * 60)
    print("PHASE 1: Baseline ECC + LSB Steganography")
    print("=" * 60)
    
    # Setup
    os.makedirs("keys", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # 1. Generate Keys
    print("\n[1] Generating ECC keypairs...")
    start_time = time.time()
    sender_priv, sender_pub = ECCKeyManager.generate_keypair("keys/sender")
    receiver_priv, receiver_pub = ECCKeyManager.generate_keypair("keys/receiver")
    print(f"✓ Keys generated in {(time.time() - start_time)*1000:.2f}ms")
    
    # 2. Prepare Message
    message = "This is a secret message for Phase 1 baseline testing!"
    print(f"\n[2] Original message: {message}")
    print(f"    Message hash: {BaselineBenchmark.hash_data(message)[:16]}...")
    
    # 3. Encryption
    print("\n[3] Encrypting message...")
    mem_before = BaselineBenchmark.measure_memory()
    start_time = time.time()
    
    # Derive shared key
    shared_key = ECCCrypto.derive_shared_key(sender_priv, receiver_pub)
    encrypted_msg = ECCCrypto.encrypt_message(message, shared_key)
    
    encrypt_time = (time.time() - start_time) * 1000
    print(f"✓ Encrypted in {encrypt_time:.2f}ms")
    print(f"    Ciphertext size: {len(encrypted_msg)} bytes")
    
    # 4. Embedding
    print("\n[4] Embedding into image...")
    # Create a test image if none exists
    if not os.path.exists("test_image.png"):
        test_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        Image.fromarray(test_img).save("test_image.png")
        print("    ℹ Created test image (512x512)")
    
    start_time = time.time()
    LSBSteganography.embed_data("test_image.png", encrypted_msg, "output/stego_phase1.png")
    embed_time = (time.time() - start_time) * 1000
    mem_after = BaselineBenchmark.measure_memory()
    
    print(f"✓ Embedded in {embed_time:.2f}ms")
    print(f"    Memory used: {mem_after - mem_before:.2f}MB")
    
    # 5. Extraction and Decryption
    print("\n[5] Extracting and decrypting...")
    start_time = time.time()
    
    extracted_data = LSBSteganography.extract_data("output/stego_phase1.png")
    receiver_shared_key = ECCCrypto.derive_shared_key(receiver_priv, sender_pub)
    decrypted_msg = ECCCrypto.decrypt_message(extracted_data, receiver_shared_key)
    
    decode_time = (time.time() - start_time) * 1000
    
    print(f"✓ Decoded in {decode_time:.2f}ms")
    print(f"    Decrypted message: {decrypted_msg}")
    
    # 6. Validation
    print("\n[6] Validation:")
    original_hash = BaselineBenchmark.hash_data(message)
    decoded_hash = BaselineBenchmark.hash_data(decrypted_msg)
    
    if original_hash == decoded_hash:
        print(f"✓ PASS: Message integrity verified")
        print(f"    Hash match: {original_hash[:16]}...")
    else:
        print(f"✗ FAIL: Hash mismatch!")
    
    # 7. Summary
    print("\n" + "=" * 60)
    print("PHASE 1 BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Encryption time:  {encrypt_time:.2f}ms")
    print(f"Embedding time:   {embed_time:.2f}ms")
    print(f"Decoding time:    {decode_time:.2f}ms")
    print(f"Total time:       {encrypt_time + embed_time + decode_time:.2f}ms")
    print(f"Memory footprint: {mem_after - mem_before:.2f}MB")
    print(f"Message size:     {len(message)} bytes")
    print(f"Payload size:     {len(encrypted_msg)} bytes")
    print("=" * 60)


if __name__ == "__main__":
    main()