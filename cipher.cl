/* cipher.cl
* by Kurt D kurtd5105@gmail.com
* Description: Contains an encryption and decryption algorithm function for the Vigenere cipher.
*/
__kernel void encrypt(__global int* plaintext, __global int* key, __global int* ciphertext){
	// Get the ID to work on its own part of the ciphertext
	uint i = get_global_id(0);
	// Don't modify characters under 32
	if(plaintext[i] > 31){
		ciphertext[i] = plaintext[i] + key[i] - 32;
		if(ciphertext[i] > 126){
			ciphertext[i] -= 95;
		}
	} else {
		ciphertext[i] = plaintext[i];
	}
}

__kernel void decrypt(__global int* plaintext, __global int* key, __global int* ciphertext){
	uint i = get_global_id(0);
	if(plaintext[i] > 31){
		ciphertext[i] = plaintext[i] - key[i] + 32;
		if(ciphertext[i] < 32){
			ciphertext[i] += 95;
		}
	} else {
		ciphertext[i] = plaintext[i];
	}
}