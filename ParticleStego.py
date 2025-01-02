import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import scipy.stats as stats

# APSO Parameters
num_particles = 50
num_iterations = 200
initial_inertia_weight = 0.9
final_inertia_weight = 0.4
cognitive_coef = 2.0
social_coef = 2.0

def calculate_image_entropy(image_array):
    """Calculate the entropy of an image."""
    histogram, _ = np.histogram(image_array, bins=256, range=(0, 256), density=True)
    entropy = stats.entropy(histogram)
    return entropy

def fitness_function(position, original_image_array, modified_image_array):
    """Fitness function balancing embedding capacity, SSIM, and entropy."""
    # Embed data into original image using the particle's 'position'
    capacity = np.sum(position)
    ssim_r = ssim(original_image_array[:, :, 0], modified_image_array[:, :, 0], data_range=modified_image_array[:, :, 0].max() - modified_image_array[:, :, 0].min())
    ssim_g = ssim(original_image_array[:, :, 1], modified_image_array[:, :, 1], data_range=modified_image_array[:, :, 1].max() - modified_image_array[:, :, 1].min())
    ssim_b = ssim(original_image_array[:, :, 2], modified_image_array[:, :, 2], data_range=modified_image_array[:, :, 2].max() - modified_image_array[:, :, 2].min())
    ssim_value = (ssim_r + ssim_g + ssim_b) / 3
    entropy_value = (calculate_image_entropy(modified_image_array[:, :, 0]) +
                     calculate_image_entropy(modified_image_array[:, :, 1]) +
                     calculate_image_entropy(modified_image_array[:, :, 2])) / 3

    alpha, beta, gamma = 0.7, 0.15, 0.15
    fitness = alpha * capacity + beta * ssim_value + gamma * entropy_value
    return fitness

def initialize_particles(image_size):
    particles = []
    for _ in range(num_particles):
        particle = {
            'position': np.random.randint(0, 2, size=image_size),
            'velocity': np.random.uniform(-1, 1, size=image_size),
            'best_position': None,
            'best_fitness': float('-inf')
        }
        particles.append(particle)
    return particles

def update_particle(particle, global_best_position, inertia_weight):
    inertia = inertia_weight * particle['velocity']
    cognitive = cognitive_coef * np.random.rand(*particle['velocity'].shape) * (particle['best_position'] - particle['position'])
    social = social_coef * np.random.rand(*particle['velocity'].shape) * (global_best_position - particle['position'])
    particle['velocity'] = inertia + cognitive + social
    particle['position'] = np.clip(particle['position'] + particle['velocity'], 0, 1)

def aps_optimization(original_image_array, message_bin_length):
    particles = initialize_particles(original_image_array.size)
    global_best_position = None
    global_best_fitness = float('-inf')
    
    for iteration in range(num_iterations):
        inertia_weight = initial_inertia_weight - (initial_inertia_weight - final_inertia_weight) * (iteration / num_iterations)
        
        for particle in particles:
            modified_image_array = np.copy(original_image_array)
            img_flat = modified_image_array.flatten()
            for i in range(min(message_bin_length, len(img_flat))):
                if particle['position'][i] == 1:
                    img_flat[i] = (img_flat[i] & ~3) | 0  # placeholder for embedding
            modified_image_array = img_flat.reshape(original_image_array.shape)
            fitness = fitness_function(particle['position'], original_image_array, modified_image_array)
            
            if fitness > particle['best_fitness']:
                particle['best_fitness'] = fitness
                particle['best_position'] = particle['position']
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle['position']
        
        for particle in particles:
            update_particle(particle, global_best_position, inertia_weight)

    return global_best_position

def embed_message(image_path, message_path, output_path, position_path='best_position.npy', length_path='length.txt'):
    with open(message_path, 'r') as f:
        message = f.read()
    message_bin = ''.join(format(ord(c), '08b') for c in message)
    message_length = len(message_bin)

    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    best_position = aps_optimization(img_array, message_length)
    
    img_flat = img_array.flatten()
    bit_index = 0
    for i in range(len(img_flat)):
        if bit_index < message_length and best_position[i] == 1:
            img_flat[i] = (img_flat[i] & ~3) | int(message_bin[bit_index:bit_index+2], 2)
            bit_index += 2

    img_array = img_flat.reshape(img_array.shape)
    Image.fromarray(img_array.astype('uint8')).save(output_path)
    np.save(position_path, best_position)
    with open(length_path, 'w') as f:
        f.write(str(len(message)))

    print(f"Message embedded successfully. Image saved as {output_path}")

def extract_message(image_path, position_path='best_position.npy', length_path='length.txt', output_message_path='extracted_message.txt'):
    # Read the message length from the file to determine how many bits to extract
    with open(length_path, 'r') as file:
        message_length = int(file.read()) * 8  # Message length in bits

    img = Image.open(image_path)
    img_array = np.array(img)
    img_flat = img_array.flatten()
    
    # Load the best positions used in embedding
    best_position = np.load(position_path)
    
    # Extract the binary data based on the best positions
    message_bin = []
    for i in range(len(img_flat)):
        if best_position[i] == 1:
            message_bin.append(format(img_flat[i] & 3, '02b'))
            if len(message_bin) * 2 >= message_length:  # Ensure we only extract as many bits as required
                break

    # Join the binary data and convert it back to text
    message_bin = ''.join(message_bin)
    message = ''.join(chr(int(message_bin[i:i+8], 2)) for i in range(0, message_length, 8))
    
    # Write the extracted message to a file
    with open(output_message_path, 'w', encoding='utf-8') as file:
        file.write(message)
    
    print(f"Extracted message saved as {output_message_path}")

# Example Usage
embed_message('Splash.png', 'secret_message3.txt', 'output_image.png')
extract_message('output_image.png')
