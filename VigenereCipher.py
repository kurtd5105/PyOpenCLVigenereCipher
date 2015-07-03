# VigenereCipher.py
# by Kurt D kurtd5105@gmail.com
# Description: A Vigenere cipher program that offloads the encryption and decryption to the GPU.

from __future__ import print_function
import pyopencl as cl
import numpy
import sys
from string import joinfields

# textToIntArray
# Description: Takes a string and turns it into an array of the decimal value of each character.
def textToIntArray(text):
	return map(ord, text)

# intArrayToString
# Description: Takes each integer from an array, casts it to a char and joins each into a string.
def intArrayToString(intArray):
	return joinfields(map(chr, intArray), "")

class Cipher:
	# init
	# Description: Loads files, converts their input to be used with PyOpenCL
	def __init__(self, shaderFile, plaintextFile, keyFile):
		print("Loading files...")
		try:
			with open(plaintextFile, 'r') as textInput:
				plaintext = textInput.read()
		except:
			print("There was an error reading {}.".format(plaintextFile))
			sys.exit(1)

		try:
			with open(keyFile, 'r') as textInput:
				key = textInput.read()
		except:
			print("There was an error reading {}.".format(keyFile))
			sys.exit(1)

		try:
			with open(shaderFile, 'r') as shaderCode:
				self.shader = shaderCode.read()
		except:
			print("There was an error reading {}.".format(shaderFile))
			sys.exit(1)

		print("Converting input...")
		plaintext_formatted = textToIntArray(plaintext)
		key_formatted = textToIntArray(key)
		print("Reformatting input for OpenCL...")
		self.plaintext = numpy.array(plaintext_formatted)
		self.key = numpy.array(key_formatted)
		# Resizes the key so that it is as long as the plaintext
		self.key = numpy.resize(self.key, (1, len(plaintext_formatted)))
		# Creates an empty array the size of the plaintext for storing the results
		self.ciphertext = numpy.empty_like(self.plaintext)

	# prepareCL
	# Description: Prepares OpenCL for GPU offloading of the Vigenere cipher computations.
	def prepareCL(self):
		print("Preparing OpenCL...")
		# Create the OpenCL context and the queue
		self.context = cl.create_some_context()
		self.queue = cl.CommandQueue(self.context)
		# Load the program and build it for use
		self.program = cl.Program(self.context, self.shader).build()

		memory = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR

		# Create buffers for the input and output numpy arrays
		self.plaintextBuffer = cl.Buffer(self.context, memory, hostbuf = self.plaintext)
		self.keyBuffer = cl.Buffer(self.context, memory, hostbuf = self.key)
		self.ciphertextBuffer = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.plaintext.nbytes)
		
	# encrypt
	# Description: Uses the GPU to perform Vigenere cipher encryption.
	def encrypt(self):
		print("Encrypting...")
		# Start encryption on GPU
		self.program.encrypt(self.queue, self.plaintext.shape, None, self.plaintextBuffer, self.keyBuffer, self.ciphertextBuffer)
		# Read the buffer's output to the ciphertext and wait for completion
		cl.enqueue_read_buffer(self.queue, self.ciphertextBuffer, self.ciphertext).wait()

	# decrypt
	# Description: Uses the GPU to perform Vigenere cipher decryption.
	def decrypt(self):
		# Start decryption on GPU
		self.program.decrypt(self.queue, self.plaintext.shape, None, self.plaintextBuffer, self.keyBuffer, self.ciphertextBuffer)
		# Read the buffer's output to the ciphertext and wait for completion
		cl.enqueue_read_buffer(self.queue, self.ciphertextBuffer, self.ciphertext).wait()

	# output
	# Description: Outputs the ciphertext to out + mode + output suffix
	def output(self, outputSuffix):
		print("Converting output...")
		temp = intArrayToString(self.ciphertext)
		print("Outputting data...")
		with open("out{}".format(sys.argv[0]) + outputSuffix, 'w') as output:
			output.write(temp)

if __name__ == '__main__':
	# Remove the cwd
	sys.argv.remove(sys.argv[0])

	if len(sys.argv) != 4:
		print("Incorrect amount of arguments. Should be program.py [e or d] [shader] [plaintext] [key].")
		sys.exit(1)

	# Create a new cipher object based on the first 3 inputs
	main = Cipher(sys.argv[1], sys.argv[2], sys.argv[3])
	main.prepareCL()
	if(sys.argv[0] == 'd'):
		main.decrypt()
	elif(sys.argv[0] == 'e'):
		main.encrypt()
	else:
		print("Incorrect first argument, enter e for encryption, d for decryption.")
		sys.exit(1)

	main.output(sys.argv[2])
	print("Done!")