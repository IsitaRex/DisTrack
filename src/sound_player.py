

def main():
    parser = argparse.ArgumentParser(description='Audio Dataset Sample Player')
    parser.add_argument('--file', type=str, default='result/res_DM_URBANSOUND8K_ConvNet_10ipc.pt',
                        help='Path to results file')
    parser.add_argument('--dataset', type=str, choices=['URBANSOUND8K', 'GTZAN', 'AUDIO_MNIST'], required=True,
                        help='Dataset type (URBANSOUND8K, GTZAN, AUDIO_MNIST)')
    parser.add_argument('--idx', type=int, default=None,
                        help='Index of sample to play')
    parser.add_argument('--class', type=int, default=None,
                        help='Class of sample to play (random sample)')
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.file)
    label_to_name = dataset_label_mappings[args.dataset]
    
    if args.idx is not None:
        # Play specific sample by index
        if args.idx < 0 or args.idx >= len(results['data'][0][0]):
            print(f"Invalid index. Must be between 0 and {len(results['data'][0][0])-1}")
            return
        
        spectrogram = results['data'][0][0][args.idx]
        label = results['data'][0][1][args.idx]
        
        print(f"Playing sample {args.idx} - {label_to_name[label]}")
        play_sample(spectrogram)
    
    elif getattr(args, 'class') is not None:
        # Play random sample of specified class
        class_idx = getattr(args, 'class')
        if class_idx < 0 or class_idx >= len(label_to_name):
            print(f"Invalid class. Must be between 0 and {len(label_to_name)-1}")
            return
        
        spectrogram, label = get_random_sample_by_class(results, class_idx)
        if spectrogram is not None:
            print(f"Playing random sample of class {class_idx} - {label_to_name[label]}")
            play_sample(spectrogram.squeeze().numpy())
    
    else:
        print("Please specify either --idx or --class to play a sample")

if __name__ == '__main__':
    main()