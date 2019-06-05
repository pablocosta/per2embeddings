from pre_process import *
from trainer import *
from model import *
import argparse
import os
import time



def main(args):





	# Load dataset
	train_file = os.path.join(args.data_path, "data_{}_{}_{}_{}.json".format(args.dataset, args.src_lang,
																			 args.trg_lang, args.max_len))
	val_file = os.path.join(args.data_path, "data_dev_{}_{}_{}.json".format(args.src_lang, args.trg_lang, args.max_len))

	start_time = time.time()
	if os.path.isfile(train_file) and os.path.isfile(val_file):
		print ("Loading data..")
		dp = DataPreprocessor()
		train_dataset, val_dataset, vocabs = dp.load_data(train_file, val_file)
	else:
		print ("Preprocessing data..")
		dp = DataPreprocessor()
		train_dataset, val_dataset, vocabs = dp.preprocess(args.train_path, args.val_path, train_file, val_file,
														   args.src_lang, args.trg_lang, args.per_lang, args.max_len)

	print ("Elapsed Time: %1.3f \n" % (time.time() - start_time))

	print ("=========== Data Stat ===========")
	print ("Train: ", len(train_dataset))
	print ("val: ", len(val_dataset))
	print ("=================================")

	train_loader = data.BucketIterator(dataset=train_dataset, batch_size=args.batch_size,
	                                   repeat=False, shuffle=True, sort_within_batch=True,
	                                   sort_key=lambda x: len(x.src))
	val_loader = data.BucketIterator(dataset=val_dataset, batch_size=args.batch_size,
	                                 repeat=False, shuffle=True, sort_within_batch=True,
	                                 sort_key=lambda x: len(x.src))

	trainer = Trainer(train_loader, val_loader, vocabs, args)
	trainer.train_iters()



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	"""
		Testar com personality embedding
	"""
	# Language setting
	parser.add_argument('--dataset', type=str, default='textgeneration')
	parser.add_argument('--src_lang', type=str, default='description.pt')
	parser.add_argument('--trg_lang', type=str, default='caption.pt')
	parser.add_argument('--per_lang', type=str, default='personality.pt')
	parser.add_argument('--max_len', type=int, default=100)

	# Model hyper-parameters
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--grad_clip', type=float, default=2)
	parser.add_argument('--num_layer', type=int, default=2)

	#embedding dimension and hideen dimension must be equals
	parser.add_argument('--embed_dim', type=int, default=600)
	parser.add_argument('--hidden_dim', type=int, default=600)
	parser.add_argument('--dropout', type=float, default=0.5)
	parser.add_argument('--attn_model', type=str, default="general")
	parser.add_argument('--decoder_lratio', type=float, default=5.0)
	parser.add_argument('--early_stoping', type=int, default=20)
	parser.add_argument('--evaluate_every', type=int, default=100)


	# Training setting
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--num_epoch', type=int, default=2650)
	parser.add_argument('--teacher_forcing', type=float, default=1.0)

	# Path
	parser.add_argument('--data_path', type=str, default='./data/')
	parser.add_argument('--train_path', type=str, default='./data/training/')
	parser.add_argument('--val_path', type=str, default='./data/dev/')

	# Dir.
	parser.add_argument('--log', type=str, default='log')
	parser.add_argument('--sample', type=str, default='sample')

	# Misc.
	parser.add_argument('--gpu_num', type=int, default=0)

	args = parser.parse_args()
	print (args)
	main(args)