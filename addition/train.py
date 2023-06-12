import json
from transformer_lens import HookedTransformer, HookedTransformerConfig
import torch
import pickle
from dataclasses import asdict

template = '''{x0}{y0}<SEP>{x0}{y1}<MID>{m}{b}'''

SEP_TOKEN = 1001
MID_TOKEN = 1002

def gen_batches(data, batch_size):
    tokens = []
    for x in data:
        toks = [
            x['p0'][0],
            x['p0'][1],
            SEP_TOKEN,
            x['p1'][0],
            x['p1'][1],
            MID_TOKEN,
            x['m'],
            x['b'],
        ]
        tokens.append(toks)
    
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = torch.split(tokens, batch_size, dim=0)

    return tokens

def loss_fn(logits, y_true):
    pred_tokens = logits[:, -3:, :]
    y_true_tokens = y_true[:, -2:]
    probs = torch.log_softmax(pred_tokens, dim=-1)
    correct_probs = torch.gather(probs, 2, y_true_tokens.unsqueeze(-1)).squeeze(-1)
    return -torch.mean(correct_probs)

def save_model(model, cfg, path):
    print('Saving model to', path)

    with open(path, 'wb') as f:
        obj = {
            'model': model.state_dict(),
            'cfg': asdict(cfg),
        }
        pickle.dump(obj, f)

def main():
    train_data = json.load(open('train.json'))
    test_data = json.load(open('test.json'))

    train_data = gen_batches(train_data[:640], 256)
    test_data = gen_batches(test_data[:640], 256)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cfg = HookedTransformerConfig(
        d_model=128,
        n_layers=1,
        n_heads=2,
        d_head=64,
        n_ctx=8,
        d_vocab=1003,
        act_fn='relu',
        attn_only=True,
        device=device,
        seed=42,
        normalization_type=None,
    )

    model = HookedTransformer(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(2):
        print("#" * 50)
        print(f'Epoch {epoch}')
        print("#" * 20)
        for i, batch in enumerate(train_data):
            logits = model(batch)
            loss = loss_fn(logits, batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i%10 == 0:
                print(f'Epoch {epoch} Batch {i} Loss {loss.item()}')

        print("-" * 20)
        with torch.no_grad():
            model.eval()
            total_loss = 0
            for batch in test_data:
                logits = model(batch)
                loss = loss_fn(logits, batch)
                total_loss += loss.item()
            print(f'Epoch {epoch} avg Test Loss {total_loss / len(test_data)}')
            model.train()

        save_model(model, cfg, f'./models/model_{epoch}.pkl')
        break

if __name__ == '__main__':
    main()
