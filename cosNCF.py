import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.sparse import dok_matrix
import matplotlib.pyplot as plt
import scipy.sparse as sp

# Load data
train_data = pd.read_csv('./data/train.txt')
valid_data = pd.read_csv('./data/valid.txt')
test_data = pd.read_csv('./data/test.txt')

# For Cosine Similarity
cos_train_data = train_data

train_data = train_data.drop('ratings', axis=1)
valid_data = valid_data.drop('ratings', axis=1)
test_data = test_data.drop('ratings', axis=1)

train_data.sort_values(by=['user'], inplace=True)
train_data.sort_values(by=['item'], inplace=True)
valid_data.sort_values(by=['user'], inplace=True)
valid_data.sort_values(by=['item'], inplace=True)
test_data.sort_values(by=['user'], inplace=True)
test_data.sort_values(by=['item'], inplace=True)

"""print('train_num_users: ',len(train_data))
print('train_num_items: ',len(train_data))
print('valid_num_users: ',len(valid_data))
print('valid_num_items: ',len(valid_data))
print('test_num_users: ',len(test_data))
print('test_num_items: ',len(test_data))"""

"""# Combine unique users and items from train, valid, test data
all_users = np.unique(np.concatenate([train_data['user'], valid_data['user'], test_data['user']]))
all_items = np.unique(np.concatenate([train_data['item'], valid_data['item'], test_data['item']]))
print('Total unique users:', len(all_users))
print('Total unique items:', len(all_items))"""

train_num_users = train_data['user'].unique()
train_num_items = train_data['item'].unique()
"""valid_num_users = valid_data['user'].unique()
valid_num_items = valid_data['item'].unique()
test_num_users = test_data['user'].unique()
test_num_items = test_data['item'].unique()"""

"""print('train_num_users unique: ',len(train_num_users))
print('train_num_items unique: ',len(train_num_items))
print('valid_num_users unique: ',len(valid_num_users))
print('valid_num_items unique: ',len(valid_num_items))
print('test_num_users unique: ',len(test_num_users))
print('test_num_items unique: ',len(test_num_items))"""

#print(train_data[:10])
#print(train_num_users[:10])
#print(train_num_items[:10])
#print(valid_data[:10])
#print(valid_num_users[:10])
#print(valid_num_items[:10])
#print(test_data[:10])
#print(test_num_users[:10])
#print(test_num_items[:10])

num_users = train_data['user'].max() + 1
num_items = train_data['item'].max() + 1

train_data = train_data.values.tolist()
train_mat = sp.dok_matrix((num_users, num_items), dtype=np.float32)
for user, item in train_data:
    train_mat[user, item] = 1.0
#print(type(train_mat))
#print(train_mat)

# Convert train_num_users to a list
train_num_users_list = train_num_users.tolist()
# Negative sampling *Train only
num_neg=4
features_neg = []
for u in train_num_users_list:
    for _ in range(num_neg):
        j = np.random.randint(num_items)  # Use num_items instead of train_num_users_list
        while train_mat.get((int(u), int(j)), 0) == 1.0:
            j = np.random.randint(num_items)
        features_neg.append([int(u), int(j)])

labels_pos = [1] * len(train_data)
labels_neg = [0] * len(features_neg)

features_fill = [[u, i] for u, i in train_data] + features_neg
labels_fill = labels_pos + labels_neg

#print('features_fill',features_fill)
#print('labels_fill',labels_fill)
#print('features_fill',len(features_fill))
#print('labels_fill',len(labels_fill))
#print(train_num_users_list[:10])

class NCFData(Dataset):
    def __init__(self, features, num_item, num_neg, labels):
        self.features = features
        self.num_item = num_item
        self.num_neg = num_neg
        self.features_fill = features_fill
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.num_neg > 0:
            user, item = self.features[idx]
        elif self.num_neg <= 0:
            #print('self.features_pos: ',self.features_pos[:5])
            #print('self.labels_fill: ',self.labels[:5])
            #user, item = self.features_pos[idx]
            features = self.features.iloc[idx]
            user = features.iloc[0]
            item = features.iloc[1]
            #print('self.features_pos: ',len(self.features_pos))

        label = self.labels[idx]

        return user, item, label

#
valid_num_items = valid_data['item'].max() + 1
test_num_items = test_data['item'].max() + 1

valid_labels = [1] * len(valid_data)
test_labels = [1] * len(test_data)
#print(len(valid_data))
#print(len(valid_labels))
#print(type(valid_labels))
train_dataset = NCFData(features_fill, num_items, num_neg=4, labels=labels_fill)
valid_dataset = NCFData(valid_data, valid_num_items, num_neg=0, labels=valid_labels)
test_dataset = NCFData(test_data, test_num_items, num_neg=0, labels=test_labels)

# Define the NCF model
class NCF(nn.Module):
    def __init__(self, users, items, embedding_dim, layers):
        super(NCF, self).__init__()
        # GMF embeddings
        self.user_embedding_gmf = nn.Embedding(users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(items, embedding_dim)

        # MLP embeddings
        self.user_embedding_mlp = nn.Embedding(users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(items, embedding_dim)

        # MLP layers
        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip([2 * embedding_dim] + layers[: -1], layers)):
          self.fc_layers.append(nn.Linear(in_size, out_size))
          self.fc_layers.append(nn.ReLU())

        # Final prediction layer
        #self.output = nn.Linear(layers[-1], 1)
        #self.output = nn.Linear(layers[-1], 1)
        # embedding_dim =  8 (require 24)
        #self.output = nn.Linear(embedding_dim * len(layers) , 1)
        # embedding_dim =  16 (require 48)
        #self.output = nn.Linear(embedding_dim * 2 , 1)
        # embedding_dim =  32 (require 48)
        self.output = nn.Linear(48 , 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item):
        # GMF
        user_embedded_gmf = self.user_embedding_gmf(user)
        item_embedded_gmf = self.item_embedding_gmf(item)
        gmf_output = torch.mul(user_embedded_gmf, item_embedded_gmf)

        # MLP
        user_embedded_mlp = self.user_embedding_mlp(user)
        item_embedded_mlp = self.item_embedding_mlp(item)

        mlp_output = torch.cat((user_embedded_mlp, item_embedded_mlp), dim=1)
        for layer in self.fc_layers:
            mlp_output = layer(mlp_output)

        # Concatenate GMF and MLP outputs
        concat = torch.cat([gmf_output, mlp_output], dim=-1)

        output = self.output(concat)
        prediction = self.sigmoid(output)
        return prediction.squeeze()

# Evaluation Metrics
def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0
def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0
# non-refined metrics
def metrics(model, test_loader, topK):
    HR, NDCG = [], []

    for user, item, _ in test_loader:
        user = user.cuda()
        item = item.cuda()

        predictions = model(user, item)
        # topK index
        _, indices = torch.topk(predictions, topK)
        # topK items
        recommends = torch.take(item, indices).cpu().numpy().tolist()
        #print('recommends: ', recommends)
        gt_items = item.cpu().numpy().tolist()
        for gt_item in gt_items:
            HR.append(hit(gt_item, recommends))
            NDCG.append(ndcg(gt_item, recommends))
        """gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))"""

    return np.mean(HR), np.mean(NDCG)

# Cosine Similarity Part
from sklearn.metrics.pairwise import cosine_similarity

item_ratings_matrix = cos_train_data.pivot_table(index='user', columns='item', values='ratings', aggfunc='mean').fillna(0)

# Compute item similarity matrix based on cosine similarity of item ratings
item_similarity_matrix = cosine_similarity(item_ratings_matrix.T, dense_output=False)

def refine_recommendations(recommendations_from_ncf, topK):
    valid_items = [item for item in recommendations_from_ncf if item < item_similarity_matrix.shape[0]]
    
    # Calculate each items cosine similarity score with each other item
    similar_items_scores = [
        (item, np.mean([item_similarity_matrix[item, similar_item] for similar_item in valid_items if item != similar_item]))
        for item in valid_items
    ]

    # Sort the recommended items based on their similarity scores
    similar_items_scores.sort(key=lambda x: -x[1])

    # Return the topK refined recommendations
    refined_recommendations = [item for item, score in similar_items_scores[:topK]]
    return refined_recommendations

# refined metrics
def metrics_with_refinement(model, test_loader, topK):
    HR, NDCG = [], []

    for user, item, _ in test_loader:
        user = user.cuda()
        item = item.cuda()

        predictions = model(user, item)
        _, indices = torch.topk(predictions, topK)
        recommendations_from_ncf = torch.take(item, indices).cpu().numpy().tolist()
        #print('recommendations_from_ncf: ', recommendations_from_ncf)
        refined_recommendations = refine_recommendations(recommendations_from_ncf, topK)
        #print('refined_recommendations: ', refined_recommendations)
        current_ndcg=0.0
        gt_items = item.cpu().numpy().tolist()
        for gt_item in gt_items:
            HR.append(hit(gt_item, refined_recommendations))
            NDCG.append(ndcg(gt_item, refined_recommendations))
            current_ndcg+=ndcg(gt_item, refined_recommendations)
        #print('NDCG: ',current_ndcg)

    return np.mean(HR), np.mean(NDCG)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create DataLoader
batch_size = 256
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
embedding_dim = 32
layers = [64, 32, 16]
model = NCF(num_users, num_items, embedding_dim, layers).to(device)
criterion = nn.BCELoss()
learningRate = 0.00001
optimizer = optim.Adam(model.parameters(), lr=learningRate)
topK = 10
num_epochs = 10
# Training with KFold
from sklearn.model_selection import KFold
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

train_losses = []
valid_losses = []
valid_hrs = []
valid_ndcgs = []
valid_hrs_refined = []
valid_ndcgs_refined = []

for fold, (train_index, valid_index) in enumerate(kf.split(train_dataset)):
    print(f"\nTraining Fold {fold + 1}/{num_folds}")

    # Current fold dataloader
    train_fold = torch.utils.data.Subset(train_dataset, train_index)
    valid_fold = torch.utils.data.Subset(train_dataset, valid_index)
    
    train_loader = DataLoader(train_fold, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_fold, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # Training current fold
        for user, item, label in train_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            output = model(user, item)
            loss = criterion(output.squeeze(), label.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        train_losses.append(average_loss)

        # Validation current fold
        model.eval()
        all_labels = []
        all_predictions = []
        valid_loss = 0.0

        with torch.no_grad():
            for user, item, label in valid_loader:
                user = user.to(device)
                item = item.to(device)
                label = label.to(device)

                output = model(user, item)
                predictions = output.squeeze()
                all_labels.extend(label.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                
                loss = criterion(predictions, label.float())
                valid_loss += loss.item()

        average_valid_loss = valid_loss / len(valid_loader)
        valid_losses.append(average_valid_loss)

        # HR & NDCG without refinement
        valid_hr, valid_ndcg = metrics(model, valid_loader, topK)
        valid_hrs.append(valid_hr)
        valid_ndcgs.append(valid_ndcg)

        # HR & NDCG with refinement
        valid_hr_refined, valid_ndcg_refined = metrics_with_refinement(model, valid_loader, topK)
        valid_hrs_refined.append(valid_hr_refined)
        valid_ndcgs_refined.append(valid_ndcg_refined)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_loss:.4f}, Validation Loss: {average_valid_loss:.4f}, HR: {valid_hr:.4f}, NDCG: {valid_ndcg:.4f}, Refined HR: {valid_hr_refined:.4f}, Refined NDCG: {valid_ndcg_refined:.4f}")

# Average HR & NDCG over all folds
avg_valid_hr = np.mean(valid_hrs)
avg_valid_ndcg = np.mean(valid_ndcgs)
avg_valid_hr_refined = np.mean(valid_hrs_refined)
avg_valid_ndcg_refined = np.mean(valid_ndcgs_refined)

print(f"\nAverage Validation HR: {avg_valid_hr:.4f}, NDCG: {avg_valid_ndcg:.4f}, Refined HR: {avg_valid_hr_refined:.4f}, Refined NDCG: {avg_valid_ndcg_refined:.4f}")

# Plotting
plt.figure(figsize=(30, 15))

# Plot Training and Validation Loss
plt.subplot(2, 3, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train')
plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Validation HR
plt.subplot(2, 3, 2)
plt.plot(range(1, len(valid_hrs) + 1), valid_hrs, label='Validation HR')
plt.title('Validation HR')
plt.xlabel('Epoch')
plt.ylabel('HR')
plt.legend()

# Plot Validation NDCG
plt.subplot(2, 3, 3)
plt.plot(range(1, len(valid_ndcgs) + 1), valid_ndcgs, label='Validation NDCG')
plt.title('Validation NDCG')
plt.xlabel('Epoch')
plt.ylabel('NDCG')
plt.legend()

# Plot Validation HR (Refined)
plt.subplot(2, 3, 4)
plt.plot(range(1, len(valid_hrs_refined) + 1), valid_hrs_refined, label='Validation HR Refined')
plt.title('Validation HR Refined')
plt.xlabel('Epoch')
plt.ylabel('Refined HR')
plt.legend()

# Plot Validation NDCG (Refined)
plt.subplot(2, 3, 5)
plt.plot(range(1, len(valid_ndcgs_refined) + 1), valid_ndcgs_refined, label='Validation NDCG Refined')
plt.title('Validation NDCG Refined')
plt.xlabel('Epoch')
plt.ylabel('Refined NDCG')
plt.legend()

plt.subplots_adjust(wspace=0.5)
plt.show()

# Testing
model.eval()
all_labels = []
all_predictions = []
test_loss = 0.0
with torch.no_grad():
    for user, item, label in test_loader:
        user, item, label = user.to(device), item.to(device), label.to(device)
        output = model(user, item)
        #predictions = torch.sigmoid(output.squeeze())
        predictions = output.squeeze()
        all_labels.extend(label.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        #print('all_labels, ',len(all_labels))
        #print('all_predictions, ',len(all_predictions))

        loss = criterion(predictions, label.float())
        test_loss += loss.item()

average_test_loss = test_loss / len(test_loader)
# Calculate test metrics
#test_accuracy = accuracy_score(all_labels, np.round(all_predictions))
#test_ndcg = ndcg_score(np.array([all_labels]), np.array([all_predictions]))
test_hr, test_ndcg = metrics(model, test_loader, topK)
test_hr_refined, test_ndcg_refined = metrics_with_refinement(model, test_loader, topK)
#print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {average_test_loss:.4f}, HR: {test_hr:.4f}, NDCG: {test_ndcg:.4f}, Refined HR: {test_hr_refined:.4f}, Refined NDCG: {test_ndcg_refined:.4f}")