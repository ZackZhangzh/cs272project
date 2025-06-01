from imagededup.methods import PHash

phasher = PHash()
image_dir = "/home/zhihao/cs272project/Gradio/examples/photos"
# Generate encodings for all images in an image directory
encodings = phasher.encode_images(image_dir=image_dir)

# Find duplicates using the generated encodings
duplicates = phasher.find_duplicates(encoding_map=encodings)
print("find_duplicates:", duplicates)


from imagededup.methods import CNN

cnn_encoder = CNN()
duplicates = cnn_encoder.find_duplicates(
    image_dir=image_dir,
    min_similarity_threshold=0.85,
    scores=False,
    outfile="output/my_duplicates.json",
)

print("find_duplicates:", duplicates)


duplicates = cnn_encoder.find_duplicates_to_remove(
    image_dir=image_dir,
    min_similarity_threshold=0.85,
    outfile="output/my_duplicates_to_remove.json",
)
print("find_duplicates_to_remove:", duplicates)
