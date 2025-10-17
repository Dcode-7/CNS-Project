"""
Phase 2: Optimized Lightweight ECC Steganography
Uses ephemeral keys, streamlined payload, and memory-efficient operations
"""

import os
import time
import struct
import hashlib
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from PIL import Image
import numpy as np
import psutil


class OptimizedECCCrypto:
    """Optimized ECC operations with ephemeral keys"""
    
    def __init__(self):
        self.curve = ec.SECP256R1()
    
    def generate_ephemeral_keypair(self):
        """Generate ephemeral ECC keypair for single-use"""
        private_key = ec.generate_private_key(self.curve)
        return private_key, private_key.public_key()
    
    def derive_key_hkdf(self, private_key, peer_public_key, salt=None):
        """Optimized key derivation with optional salt"""
        shared_secret = private_key.exchange(ec.ECDH(), peer_public_key)
        
        # Use HKDF with minimal overhead
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,  # AES-256
            salt=salt,
            info=b'opt-ecc-stego'
        ).derive(shared_secret)
    
    def encrypt_aes_gcm(self, plaintext, key):
        """Encrypt using AES-GCM (built-in authentication)"""
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)  # 96-bit nonce for GCM
        
        # Encrypt and authenticate in one operation
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        
        # Return nonce + ciphertext (tag is included in ciphertext)
        return nonce + ciphertext
    
    def decrypt_aes_gcm(self, encrypted_data, key):
        """Decrypt AES-GCM data"""
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        
        return plaintext.decode('utf-8')


class OptimizedLSBEmbed:
    """Memory-efficient LSB embedding"""
    
    @staticmethod
    def create_payload(ephemeral_pubkey, ciphertext):
        """
        Construct optimized payload:
        [ephemeral_pubkey_bytes (33)] + [ciphertext_len (4)] + [ciphertext]
        """
        # Serialize ephemeral public key to compressed format
        pubkey_bytes = ephemeral_pubkey.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.CompressedPoint
        )
        
        # Pack length as 4-byte unsigned int
        length_bytes = struct.pack('>I', len(ciphertext))
        
        return pubkey_bytes + length_bytes + ciphertext
    
    @staticmethod
    def parse_payload(payload_bytes):
        """Parse payload to extract components"""
        # X962 compressed point for SECP256R1 is 33 bytes
        pubkey_bytes = payload_bytes[:33]
        length_bytes = payload_bytes[33:37]
        
        ciphertext_len = struct.unpack('>I', length_bytes)[0]
        ciphertext = payload_bytes[37:37+ciphertext_len]
        
        # Deserialize public key from compressed point (X9.62) directly
        ephemeral_pubkey = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256R1(), pubkey_bytes
        )
        
        return ephemeral_pubkey, ciphertext
    
    @staticmethod
    def embed_lsb_optimized(image_array, data_bytes):
        """
        Optimized LSB embedding with minimal memory allocation
        Embeds in-place without creating extra copies
        """
        # Convert data to bit string
        bit_string = ''.join(format(b, '08b') for b in data_bytes)
        data_len = len(bit_string)
        
        # Check capacity
        img_flat = image_array.ravel()
        if data_len > len(img_flat):
            raise ValueError(f"Data too large: {data_len} bits needed, {len(img_flat)} available")
        
        # Embed directly into flattened view (no copy)
        for i in range(data_len):
            img_flat[i] = (img_flat[i] & 0xFE) | int(bit_string[i])
        
        return image_array
    
    @staticmethod
    def extract_lsb_optimized(image_array, payload_size_bytes):
        """Extract LSB bits efficiently"""
        img_flat = image_array.ravel()
        num_bits = payload_size_bytes * 8
        
        # Extract bits
        bits = [str(img_flat[i] & 1) for i in range(num_bits)]
        bit_string = ''.join(bits)
        
        # Convert to bytes
        payload = bytearray()
        for i in range(0, len(bit_string), 8):
            byte_bits = bit_string[i:i+8]
            payload.append(int(byte_bits, 2))
        
        return bytes(payload)


class OptimizedStegoSystem:
    """Complete optimized steganography system"""
    
    def __init__(self):
        self.crypto = OptimizedECCCrypto()
        self.embed = OptimizedLSBEmbed()
    
    def encode(self, message, receiver_pubkey, cover_image_path, output_path):
        """
        Encode message into image with optimized workflow
        Returns: (encode_time_ms, memory_mb, payload_size)
        """
        mem_start = self._get_memory()
        time_start = time.perf_counter()
        
        # 1. Generate ephemeral keypair
        eph_priv, eph_pub = self.crypto.generate_ephemeral_keypair()
        
        # 2. Derive shared key with receiver
        shared_key = self.crypto.derive_key_hkdf(eph_priv, receiver_pubkey)
        
        # 3. Encrypt message
        ciphertext = self.crypto.encrypt_aes_gcm(message, shared_key)
        
        # 4. Create payload
        payload = self.embed.create_payload(eph_pub, ciphertext)
        
        # 5. Load image (memory-efficient)
        img = Image.open(cover_image_path).convert('RGB')
        img_array = np.array(img, dtype=np.uint8)
        
        # 6. Embed payload (in-place operation)
        stego_array = self.embed.embed_lsb_optimized(img_array, payload)
        
        # 7. Save stego image
        Image.fromarray(stego_array).save(output_path, 'PNG', optimize=True)
        
        time_elapsed = (time.perf_counter() - time_start) * 1000
        mem_used = self._get_memory() - mem_start
        
        return time_elapsed, mem_used, len(payload)
    
    def decode(self, stego_image_path, receiver_privkey):
        """
        Decode message from stego image
        Returns: (decoded_message, decode_time_ms)
        """
        time_start = time.perf_counter()
        
        # 1. Load stego image
        img = Image.open(stego_image_path).convert('RGB')
        img_array = np.array(img, dtype=np.uint8)
        
        # 2. Extract header to get payload size
        # First 33 bytes (ephemeral pubkey) + 4 bytes (length)
        header_bits = 37 * 8
        img_flat = img_array.ravel()
        header_bit_string = ''.join(str(img_flat[i] & 1) for i in range(header_bits))
        
        # Parse length from header
        length_start = 33 * 8
        length_bits = header_bit_string[length_start:length_start + 32]
        ciphertext_len = int(length_bits, 2)
        
        # Calculate total payload size
        total_payload_size = 37 + ciphertext_len
        
        # 3. Extract full payload
        payload = self.embed.extract_lsb_optimized(img_array, total_payload_size)
        
        # 4. Parse payload
        eph_pubkey, ciphertext = self.embed.parse_payload(payload)
        
        # 5. Derive shared key
        shared_key = self.crypto.derive_key_hkdf(receiver_privkey, eph_pubkey)
        
        # 6. Decrypt message
        message = self.crypto.decrypt_aes_gcm(ciphertext, shared_key)
        
        time_elapsed = (time.perf_counter() - time_start) * 1000
        
        return message, time_elapsed
    
    @staticmethod
    def _get_memory():
        """Get current memory usage in MB"""
        return psutil.Process().memory_info().rss / 1024 / 1024


def load_or_generate_keys():
    """Load existing keys or generate new ones"""
    if os.path.exists("keys/receiver_private.pem"):
        print("Loading existing receiver keys...")
        with open("keys/receiver_private.pem", "rb") as f:
            receiver_priv = serialization.load_pem_private_key(f.read(), password=None)
        with open("keys/receiver_public.pem", "rb") as f:
            receiver_pub = serialization.load_pem_public_key(f.read())
    else:
        print("Generating new receiver keys...")
        receiver_priv = ec.generate_private_key(ec.SECP256R1())
        receiver_pub = receiver_priv.public_key()
        
        os.makedirs("keys", exist_ok=True)
        with open("keys/receiver_private.pem", "wb") as f:
            f.write(receiver_priv.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        with open("keys/receiver_public.pem", "wb") as f:
            f.write(receiver_pub.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))
    
    return receiver_priv, receiver_pub


def main():
    """Main demonstration of Phase 2 optimized system"""
    print("=" * 70)
    print("PHASE 2: Optimized Lightweight ECC Steganography")
    print("=" * 70)
    
    # Setup
    os.makedirs("output", exist_ok=True)
    
    # Load/generate receiver keys
    print("\n[1] Setting up keys...")
    receiver_priv, receiver_pub = load_or_generate_keys()
    print("✓ Receiver keys ready")
    
    # Create test image if needed
    if not os.path.exists("test_image.png"):
        print("\n[2] Creating test image (512x512)...")
        test_img = np.random.randint(100, 200, (512, 512, 3), dtype=np.uint8)
        Image.fromarray(test_img).save("test_image.png")
        print("✓ Test image created")
    
    # Initialize system
    stego = OptimizedStegoSystem()
    
    # Test message
    message = "Optimized ECC steganography with ephemeral keys and AES-GCM!"
    print(f"\n[3] Original message:")
    print(f"    \"{message}\"")
    print(f"    Hash: {hashlib.sha256(message.encode()).hexdigest()[:16]}...")
    
    # Encode
    print("\n[4] Encoding with optimized workflow...")
    encode_time, memory_used, payload_size = stego.encode(
        message,
        receiver_pub,
        "test_image.png",
        "output/stego_phase2.png"
    )
    
    print(f"✓ Encoding complete")
    print(f"    Time: {encode_time:.2f}ms")
    print(f"    Memory: {memory_used:.2f}MB")
    print(f"    Payload: {payload_size} bytes")
    print(f"    Overhead: {payload_size - len(message)} bytes")
    
    # Decode
    print("\n[5] Decoding from stego image...")
    decoded_msg, decode_time = stego.decode(
        "output/stego_phase2.png",
        receiver_priv
    )
    
    print(f"✓ Decoding complete")
    print(f"    Time: {decode_time:.2f}ms")
    print(f"    Decoded: \"{decoded_msg}\"")
    
    # Validation
    print("\n[6] Validation:")
    original_hash = hashlib.sha256(message.encode()).hexdigest()
    decoded_hash = hashlib.sha256(decoded_msg.encode()).hexdigest()
    
    if original_hash == decoded_hash:
        print(f"✓ PASS: Message integrity verified")
        print(f"    Hash: {original_hash[:32]}...")
    else:
        print(f"✗ FAIL: Hash mismatch!")
        print(f"    Original: {original_hash[:32]}...")
        print(f"    Decoded:  {decoded_hash[:32]}...")
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 2 BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Encoding time:       {encode_time:.2f}ms")
    print(f"Decoding time:       {decode_time:.2f}ms")
    print(f"Total time:          {encode_time + decode_time:.2f}ms")
    print(f"Memory footprint:    {memory_used:.2f}MB")
    print(f"Payload efficiency:  {len(message)}/{payload_size} = {len(message)/payload_size*100:.1f}%")
    print(f"Ephemeral overhead:  33 bytes (compressed EC point)")
    print("=" * 70)
    
    print("\n✓ Phase 2 implementation complete!")
    print("  Next: Run Phase 3 for comparative benchmarking")


if __name__ == "__main__":
    main()