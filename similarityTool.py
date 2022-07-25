from sentence_transformers import SentenceTransformer, util

user_input1 = str(input())
user_input2 = str(input())

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#Compute embedding for both lists
embedding_1= model.encode(user_input1, convert_to_tensor=True)
embedding_2 = model.encode(user_input2, convert_to_tensor=True)

util.pytorch_cos_sim(embedding_1, embedding_2)