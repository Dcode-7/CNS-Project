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
        """Generate ECC keypair using SECP256R1s"""
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
    """Phase 1: ECC + LSB Steganography (Detailed flow with full metrics)"""
    print("=" * 70)
    print("PHASE 1: Baseline ECC + LSB Steganography (Full Metrics & Message Flow)")
    print("=" * 70)

    # Setup directories
    os.makedirs("keys", exist_ok=True)
    os.makedirs("stego_img", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)

    image_dir = os.path.join("dataset", "images")
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("⚠️ No images found in dataset/. Add images and rerun.")
        return
    print(f"Found {len(image_files)} image(s) in dataset folder.\n")

    # Message to embed
    message = "This is a secret message for Phase 1 dataset testing!"
    msg_size = len(message.encode())
    print(f"[INFO] Using message: {message}")
    print(f"       Message size: {msg_size} bytes")
    print(f"       Message hash: {BaselineBenchmark.hash_data(message)[:16]}...\n")

    # Metrics accumulators
    total_encrypt = total_embed = total_decode = total_time = 0.0
    total_mem = 0.0
    total_msg_size = total_payload = 0

    for idx, image_name in enumerate(image_files[:5], start=1):  # process first 5 for demo
        print("=" * 60)
        print(f"[{idx}/{len(image_files)}] Processing image: {image_name}")
        print("=" * 60)

        image_path = os.path.join(image_dir, image_name)
        output_path = os.path.join("stego_img", f"stego_{image_name.split('.')[0]}.png")

        # 1️⃣ Generate ECC keys
        start_time = time.time()
        sender_priv, sender_pub = ECCKeyManager.generate_keypair(f"keys/sender_{idx}")
        receiver_priv, receiver_pub = ECCKeyManager.generate_keypair(f"keys/receiver_{idx}")
        key_time = (time.time() - start_time) * 1000
        print(f"✓ ECC keypairs generated in {key_time:.2f}ms")

        # 2️⃣ Encrypt message
        print("\n[Encrypting message...]")
        print(f"Original message: {message}")
        mem_before = BaselineBenchmark.measure_memory()
        start_time = time.time()
        shared_key = ECCCrypto.derive_shared_key(sender_priv, receiver_pub)
        encrypted_msg = ECCCrypto.encrypt_message(message, shared_key)
        encrypt_time = (time.time() - start_time) * 1000
        mem_after = BaselineBenchmark.measure_memory()
        payload_size = len(encrypted_msg)
        print(f"Encrypted message (first 80 chars): {encrypted_msg[:80]}")
        print(f"✓ Encrypted in {encrypt_time:.2f}ms | Payload size: {payload_size} bytes")

        # 3️⃣ Embed ciphertext in image
        print("\n[Embedding data in image...]")
        start_time = time.time()
        try:
            LSBSteganography.embed_data(image_path, encrypted_msg, output_path)
            embed_time = (time.time() - start_time) * 1000
            print(f"✓ Embedded successfully in {embed_time:.2f}ms → {output_path}")
        except Exception as e:
            print(f"✗ Embedding failed for {image_name}: {e}")
            continue

        # 4️⃣ Extract and decrypt
        print("\n[Extracting and decrypting data...]")
        start_time = time.time()
        extracted_data = LSBSteganography.extract_data(output_path)
        print(f"Extracted message (first 80 chars): {extracted_data[:80]}")
        receiver_shared_key = ECCCrypto.derive_shared_key(receiver_priv, sender_pub)
        decrypted_msg = ECCCrypto.decrypt_message(extracted_data, receiver_shared_key)
        decode_time = (time.time() - start_time) * 1000
        print(f"Decrypted message: {decrypted_msg}")
        print(f"✓ Decrypted in {decode_time:.2f}ms")

        # 5️⃣ Verify integrity
        total_processing_time = key_time + encrypt_time + embed_time + decode_time
        mem_used = mem_after - mem_before
        if BaselineBenchmark.hash_data(message) == BaselineBenchmark.hash_data(decrypted_msg):
            print("✅ PASS: Message integrity verified.")
        else:
            print("⚠️ FAIL: Message hash mismatch!")

        # 6️⃣ Per-image summary
        print("-" * 60)
        print(f"Encryption time:    {encrypt_time:.2f}ms")
        print(f"Embedding time:     {embed_time:.2f}ms")
        print(f"Decoding time:      {decode_time:.2f}ms")
        print(f"Total processing:   {total_processing_time:.2f}ms")
        print(f"Memory used:        {mem_used:.2f}MB")
        print(f"Message size:       {msg_size} bytes")
        print(f"Payload size:       {payload_size} bytes")
        print(f"Stego image saved:  {output_path}")
        print("-" * 60 + "\n")

        # accumulate metrics
        total_encrypt += encrypt_time
        total_embed += embed_time
        total_decode += decode_time
        total_mem += mem_used
        total_time += total_processing_time
        total_msg_size += msg_size
        total_payload += payload_size

    # ✅ Final summary
    n = len(image_files[:5])
    print("\n" + "=" * 70)
    print("PHASE 1 BENCHMARK SUMMARY (All Dataset Images)")
    print("=" * 70)
    print(f"Total images processed: {n}")
    print(f"Avg Encryption time:   {total_encrypt/n:.2f}ms")
    print(f"Avg Embedding time:    {total_embed/n:.2f}ms")
    print(f"Avg Decoding time:     {total_decode/n:.2f}ms")
    print(f"Avg Total time:        {total_time/n:.2f}ms per image")
    print(f"Avg Memory footprint:  {total_mem/n:.2f}MB")
    print(f"Avg Message size:      {total_msg_size/n:.2f} bytes")
    print(f"Avg Payload size:      {total_payload/n:.2f} bytes")
    print("=" * 70)




if __name__ == "__main__":
    main()