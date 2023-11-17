from src.config import parse_args
from data_loader import get_data_loader
from src.dcgan import DCGAN


def main(args):
    model = DCGAN(args)

    # Load datasets to train and test loaders
    train_loader, test_loader = get_data_loader(args)

    # Start model training
    if args.is_train == 'True':
        model.train(train_loader)

    # start evaluating on test data
    else:
        model.evaluate(test_loader, args.load_D, args.load_G)
        for i in range(50):
           model.generate_latent_walk(i)





if __name__ == '__main__':
    args = parse_args()
    print(args.cuda)
    main(args)