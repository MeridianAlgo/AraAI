"""
API Key Encryption Module

Provides encryption and secure storage for API keys and sensitive data.
Uses Fernet (symmetric encryption) from cryptography library.
"""

import os
import base64
import hashlib
from typing import Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pathlib import Path


class KeyEncryption:
    """Handles encryption and decryption of API keys and sensitive data"""

    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize key encryption

        Args:
            master_key: Master encryption key (if None, generates or loads from env)
        """
        if master_key:
            self.master_key = master_key.encode()
        else:
            self.master_key = self._get_or_create_master_key()

        self.fernet = self._create_fernet()

    def _get_or_create_master_key(self) -> bytes:
        """
        Get or create master encryption key

        Returns:
            Master key bytes
        """
        # Try to get from environment variable
        env_key = os.getenv("ARA_MASTER_KEY")
        if env_key:
            return env_key.encode()

        # Try to load from file
        key_file = Path.home() / ".ara" / "master.key"
        if key_file.exists():
            with open(key_file, "rb") as f:
                return f.read()

        # Generate new key
        key = Fernet.generate_key()

        # Save to file
        key_file.parent.mkdir(parents=True, exist_ok=True)
        with open(key_file, "wb") as f:
            f.write(key)

        # Set restrictive permissions (owner read/write only)
        os.chmod(key_file, 0o600)

        return key

    def _create_fernet(self) -> Fernet:
        """
        Create Fernet cipher from master key

        Returns:
            Fernet cipher instance
        """
        # If master key is not in Fernet format, derive it
        if len(self.master_key) != 44:  # Fernet keys are 44 bytes base64
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"ara_ai_salt",  # In production, use random salt
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        else:
            key = self.master_key

        return Fernet(key)

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt plaintext string

        Args:
            plaintext: String to encrypt

        Returns:
            Encrypted string (base64 encoded)
        """
        if not plaintext:
            return ""

        encrypted_bytes = self.fernet.encrypt(plaintext.encode())
        return encrypted_bytes.decode()

    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt ciphertext string

        Args:
            ciphertext: Encrypted string (base64 encoded)

        Returns:
            Decrypted plaintext string

        Raises:
            ValueError: If decryption fails
        """
        if not ciphertext:
            return ""

        try:
            decrypted_bytes = self.fernet.decrypt(ciphertext.encode())
            return decrypted_bytes.decode()
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")

    def hash_api_key(self, api_key: str) -> str:
        """
        Create a one-way hash of API key for storage/comparison

        Args:
            api_key: API key to hash

        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(api_key.encode()).hexdigest()

    def encrypt_api_key(self, api_key: str) -> str:
        """
        Encrypt API key for secure storage

        Args:
            api_key: API key to encrypt

        Returns:
            Encrypted API key
        """
        return self.encrypt(api_key)

    def decrypt_api_key(self, encrypted_key: str) -> str:
        """
        Decrypt API key from storage

        Args:
            encrypted_key: Encrypted API key

        Returns:
            Decrypted API key
        """
        return self.decrypt(encrypted_key)

    def mask_api_key(self, api_key: str, visible_chars: int = 4) -> str:
        """
        Mask API key for display (show only last N characters)

        Args:
            api_key: API key to mask
            visible_chars: Number of characters to show at end

        Returns:
            Masked API key (e.g., "****abcd")
        """
        if len(api_key) <= visible_chars:
            return "*" * len(api_key)

        return "*" * (len(api_key) - visible_chars) + api_key[-visible_chars:]

    @staticmethod
    def generate_api_key(length: int = 32) -> str:
        """
        Generate a secure random API key

        Args:
            length: Length of API key

        Returns:
            Random API key string
        """
        # Generate random bytes
        random_bytes = os.urandom(length)

        # Convert to base64 and remove padding
        api_key = base64.urlsafe_b64encode(random_bytes).decode().rstrip("=")

        return api_key[:length]

    def rotate_master_key(self, new_master_key: str) -> None:
        """
        Rotate the master encryption key

        Note: This requires re-encrypting all stored encrypted data

        Args:
            new_master_key: New master key
        """
        # Create new Fernet with new key
        old_fernet = self.fernet

        self.master_key = new_master_key.encode()
        self.fernet = self._create_fernet()

        # Note: Caller must re-encrypt all stored data
        # This method just updates the encryption instance

    @staticmethod
    def secure_compare(a: str, b: str) -> bool:
        """
        Timing-safe string comparison to prevent timing attacks

        Args:
            a: First string
            b: Second string

        Returns:
            True if strings are equal
        """
        if len(a) != len(b):
            return False

        result = 0
        for x, y in zip(a, b):
            result |= ord(x) ^ ord(y)

        return result == 0
