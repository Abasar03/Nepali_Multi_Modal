def generate_aligned_embeddings():
    device = Configuration.device

    BATCH_SIZE = 8

    torch.cuda.empty_cache()
    try:
        contrastive_model = ContrastiveModel().to(device)
        contrastive_model.load_state_dict(
            torch.load('/content/drive/MyDrive/Minor_project/contrastive_model.pt',
                      map_location=device)
        )
        contrastive_model.eval()

        fusion_model = MultiModalFusion().to(device)
        fusion_model.load_state_dict(
            torch.load('/content/drive/MyDrive/Minor_project/small_fused_v3.pt',
                      map_location=device)
        )
        fusion_model.eval()
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

    df = pd.read_csv('captions.csv')
    tokenizer = AutoTokenizer.from_pretrained(Configuration.text_tokenizer)

    Configuration.num_workers = 0
    Configuration.batch_size = BATCH_SIZE

    try:
        dataloader = build_loaders(df, tokenizer, mode='train')
    except Exception as e:
        print(f"Error building dataloader: {str(e)}")
        raise

    results = []
    embedding_shapes = set()


    total_batches = len(dataloader)
    print(f"Total batches to process: {total_batches}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating Embeddings")):
            try:
                captions = batch.pop('caption')
                device_batch = {k: v.to(device) for k, v in batch.items()}

                image_features = contrastive_model.image_encoder(device_batch['image'])
                text_features = contrastive_model.text_encoder(
                    input_ids=device_batch['input_ids'],
                    attention_mask=device_batch['attention_mask']
                )
                image_features = contrastive_model.image_projection(image_features)
                text_features = contrastive_model.text_projection(text_features)

                fused_embeddings = fusion_model(image_features, text_features)
                print(f'shape before {fused_embeddings.shape}')

                fused_embeddings = fused_embeddings.squeeze(0)


                if batch_idx == 0:
                    print(f"First batch shapes:")
                    print(f"Image features: {image_features.shape}")
                    print(f"Text features: {text_features.shape}")
                    print(f"Fused embeddings: {fused_embeddings.shape}")



                for i in range(fused_embeddings.shape[0]):
                    embedding = fused_embeddings[i].cpu().numpy()
                    embedding_shapes.add(embedding.shape)
                    results.append({
                        'caption': captions[i],
                        'embedding': embedding
                    })

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                raise e

    result_df = pd.DataFrame(results)



    print("\nEmbedding shape analysis:")
    print(f"Number of unique embedding shapes found: {len(embedding_shapes)}")
    # for shape in embedding_shapes:
        # print(f"Shape: {shape}")


    first_shape = result_df['embedding'][0].shape
    invalid_indices = []
    for idx, emb in enumerate(result_df['embedding']):
        if emb.shape != first_shape:
            invalid_indices.append(idx)
            print(f"Mismatched shape at index {idx}: Expected {first_shape}, got {emb.shape}")

    if invalid_indices:
        print(f"\nRemoving {len(invalid_indices)} invalid embeddings")
        result_df = result_df.drop(invalid_indices)

    if len(result_df) > 0:
        torch.save({
            'captions': result_df['caption'].values,
            'embeddings': np.stack(result_df['embedding'].values),
            'metadata': {
                'num_samples': len(result_df),
                'embedding_dim': result_df['embedding'][0].shape[0],
                'generation_date': pd.Timestamp.now().isoformat()
            }
        }, '/content/drive/MyDrive/Minor_project/random_aligned_v3.pt')
        print(f"\nSuccessfully saved {len(result_df)} embeddings")
    else:
        raise ValueError("No valid embeddings to save")


    print(f"Number of unique captions: {len(result_df['caption'].unique())}")
    print(f"Total number of embeddings: {len(result_df)}")
    print(f"Shape of first embedding: {result_df['embedding'][0].shape}")

    embedding_dims = [emb.shape[0] for emb in result_df['embedding']]
    assert all(dim == 1024 for dim in embedding_dims), "Not all embeddings are 1024-dimensional"

    if len(result_df['caption'].unique()) != len(result_df):
        print("Warning: There are duplicate captions!")

    return result_df

if __name__ == "__main__":
  embedding_df = generate_aligned_embeddings()
