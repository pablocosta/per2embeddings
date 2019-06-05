from bleu import *
from helper import tensor2np, AverageMeter, get_word_embeddings
from pre_process import *
import torch.nn as nn
from model import *
import torch.optim as optim
import random
import itertools
import pandas as pd
from rouge import Rouge


class Trainer(object):
    def __init__(self, train_loader, val_loader, vocabs, args):

        self.use_cuda = True
        self.max_length = args.max_len

        # Data Loader
        self.train_loader = train_loader
        self.val_loader   = val_loader
        # Hyper-parameters
        self.lr             = args.lr
        self.grad_clip      = args.grad_clip
        self.embed_dim      = args.embed_dim
        self.hidden_dim     = args.hidden_dim
        self.num_layer      = args.num_layer
        self.dropout        = args.dropout
        self.attn_model     = args.attn_model
        self.decoder_lratio = args.decoder_lratio
        self.teacher_forcing= args.teacher_forcing
        self.nepoch_no_imprv= args.early_stoping
        self.evaluate_every = args.evaluate_every

        # Training setting
        self.batch_size     = args.batch_size
        self.num_epoch      = args.num_epoch
        self.iter_per_epoch = len(train_loader)

        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if USE_CUDA else "cpu")

        self.build_model(vocabs)
        self.log_path = os.path.join('./logs/' + args.log)

    def build_model(self, vocabs):
        self.src_vocab = vocabs['src_vocab']
        self.trg_vocab = vocabs['trg_vocab']
        self.per_vocab = vocabs['per_vocab']
        self.src_inv_vocab = vocabs['src_inv_vocab']
        self.trg_inv_vocab = vocabs['trg_inv_vocab']
        self.per_inv_vocab = vocabs['per_inv_vocab']

        self.trg_soi = self.trg_vocab[SOS_WORD]
        self.PAD_word = self.src_vocab[PAD_WORD]

        self.src_nword = len(self.src_vocab)
        self.trg_nword = len(self.trg_vocab)
        self.per_nword = len(self.per_vocab)


        print('Building encoder and decoder ...')
        # Initialize word embeddings
        embedding_trg = nn.Embedding(self.trg_nword, self.embed_dim)
        embedding_src = nn.Embedding(self.src_nword, self.embed_dim)
        embedding_per = nn.Embedding(self.per_nword, 600)



        #if loadFilename:
        w_trg = get_word_embeddings(600, "./data/embeddings/pt.txt", self.trg_vocab)
        w_src = get_word_embeddings(600, "./data/embeddings/pt.txt", self.src_vocab)
        embedding_trg.from_pretrained(torch.FloatTensor(w_trg))
        embedding_src.from_pretrained(torch.FloatTensor(w_src))



        # Initialize encoder & decoder models
        self.encoder = EncoderRNN(self.hidden_dim, embedding_src, embedding_per, self.num_layer, self.dropout)
        self.decoder = LuongAttnDecoderRNN(self.attn_model, embedding_trg, self.hidden_dim, self.trg_nword, self.num_layer, self.dropout)



        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()


        # set the criterion and optimizer
        self.encoder_optimizer = optim.Adam(filter(lambda x: x.requires_grad, self.encoder.parameters()),
                                      lr=self.lr)
        self.decoder_optimizer = optim.Adam(filter(lambda x: x.requires_grad, self.decoder.parameters()),
                                      lr=self.lr * self.decoder_lratio)

        self.criterion = nn.NLLLoss()

        print(self.encoder)
        print(self.decoder)
        print(self.criterion)
        print(self.encoder_optimizer)
        print(self.decoder_optimizer)
    def get_mask(self, output_batch):
        #pad value is one
        mask = binaryMatrix(output_batch, self.PAD_word)
        mask = torch.ByteTensor(mask)
        return mask

    def get_sentence(self, sentence, side):
        def _eos_parsing(sentence):
            if EOS_WORD in sentence:
                return sentence[:sentence.index(EOS_WORD) + 1]
            else:
                return sentence

        # index sentence to word sentence
        if side == 'trg':

            sentence = [self.trg_inv_vocab[x] for x in sentence]

        else:
            sentence = [self.src_inv_vocab[x] for x in sentence]



        return _eos_parsing(sentence)


        return torch.LongTensor(pad_vector)
    def train(self, input_variable, target_variable, per_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length):

        # Zero gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        lengths = []
        for input_ in input_variable:
            ctr = 0
            for ele in input_:
                if ele == self.PAD_word:
                    break
                ctr += 1
            lengths.append(ctr)



        mask = self.get_mask(target_variable)


        # Set device options
        input_variable = input_variable.t().to(self.device)
        per_variable   = per_variable.t().to(self.device)
        lengths = torch.tensor(lengths).to(self.device)
        target_variable = target_variable.t().to(self.device)
        mask = mask.t().to(self.device)

        """
        temos que checar o tamanhoo dp padding e fazer o padding na mão...
        RuntimeError: Expected
        hidden
        size(2, 40, 100), got(2, 3, 100)
        """
        # Initialize variables
        loss = 0
        print_losses = []
        n_totals = 0

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = encoder(input_variable, lengths, per_variable)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[self.trg_soi for _ in range(target_variable.size()[1])]])
        decoder_input = decoder_input.to(self.device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:decoder.n_layers]

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < self.teacher_forcing else False

        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(max_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forcing: next input is current target
                decoder_input = target_variable[t].view(1, -1)
                # Calculate and accumulate loss
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t], self.device)
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        else:
            for t in range(max_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                decoder_input = decoder_input.to(self.device)
                # Calculate and accumulate loss
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t], self.device)
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal

        # Perform backpropatation
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), self.grad_clip)
        _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), self.grad_clip)

        # Adjust model weights
        encoder_optimizer.step()
        decoder_optimizer.step()







        return sum(print_losses) / n_totals


    def train_iters(self):
        # Initialize search module
        searcher = GreedySearchDecoder(self.encoder,
                                       self.decoder, self.device)
        start = time.time()
        nepoch_no_imprv = 0
        self.train_loss = AverageMeter()
        total_loss      = []
        total_bleu    = []
        total_rouge_l = []
        total_rouge_1 = []
        total_rouge_2 = []
        self.best_bleu  = .0
        for epoch in tqdm(range(self.num_epoch)):
            # Ensure dropout layers are in train mode
            self.encoder.train()
            self.decoder.train()

            print_loss_total = 0
            epoch_loss       = []
            for i_train, batch in enumerate(self.train_loader):
                src_input = batch.src[0]
                per_input = batch.per[0]
                src_length = batch.src[1]
                trg_input = batch.trg[0][:, :-1]
                trg_length = batch.trg[1]

                max_target_lenght = max([len(ele) for ele in trg_input.tolist()])
                max_src_lenght = max([len(ele) for ele in src_input.tolist()])


                loss = self.train(src_input, trg_input, per_input, self.encoder,
                             self.decoder, self.encoder_optimizer, self.decoder_optimizer, self.criterion, max_target_lenght)
                self.train_loss.update(loss)
                epoch_loss.append(loss)
                print_loss_total += loss
            total_loss.append(np.average(epoch_loss))

            ######################################################################
            # Run the overfitting
            # ~~~~~~~~~~~~~~

            if (epoch % self.evaluate_every == 0) and (epoch != 0):
                # Ensure dropout layers are in evaluation mode
                self.encoder.eval()
                self.decoder.eval()
                self.val_bleu    = AverageMeter()
                self.val_rouge_1 = AverageMeter()
                self.val_rouge_2 = AverageMeter()
                self.val_rouge_l = AverageMeter()
                epoch_bleu = []
                epoch_rouge_l = []
                epoch_rouge_1 = []
                epoch_rouge_2 = []
                print("evaluating")
                for i, batch in enumerate(self.val_loader):
                    src_input = batch.src[0]
                    per_input = batch.per[0]
                    trg_output = batch.trg[0][:,:]
                    preds = []

                    for src_input_, per_input_ in zip(src_input, per_input):

                        lengths = []
                        ctr = 0
                        for ele in src_input_:
                            if ele == self.PAD_word:
                                break
                            ctr += 1
                        lengths.append(ctr)

                        src_input_ = torch.LongTensor([src_input_.tolist()]).transpose(0,1).to(self.device)
                        per_input_ = torch.LongTensor([per_input_.tolist()]).transpose(0,1).to(self.device)
                        lengths = torch.tensor(lengths).to(self.device)
                        #trg_output_ = torch.LongTensor([trg_output_.tolist()]).transpose(0,1).to(self.device)

                        pred, scores = searcher(src_input_, per_input_, lengths, self.max_length, self.trg_soi)
                        preds.append(pred)

                    # Compute BLEU and ROUGFE score and Loss
                    pred_sents = []
                    trg_sents  = []
                    pred_STR   = []
                    trg_STR    = []

                    print(trg_output.t().size())
                    for j in range(trg_output.t().size()[1]):

                        pred_sent = self.get_sentence(tensor2np(preds[j]), 'trg')
                        trg_sent = self.get_sentence(tensor2np(trg_output[j]), 'trg')
                        pred_sents.append(pred_sent)
                        pred_STR.append(" ".join(pred_sent))
                        trg_sents.append(trg_sent)
                        trg_STR.append(" ".join(trg_sent))


                    rouge = Rouge()
                    hyps, refs = map(list, zip(*[[d[0], d[1]] for d in [pred_STR, trg_STR]]))

                    rouge_scores = rouge.get_scores(hyps, refs, avg=True)

                    bleu_score = get_bleu(pred_sents, trg_sents)
                    epoch_bleu.append(bleu_score)

                    epoch_rouge_1.append(rouge_scores["rouge-1"]["f"])
                    epoch_rouge_2.append(rouge_scores["rouge-2"]["f"])
                    epoch_rouge_l.append(rouge_scores["rouge-l"]["f"])

                    self.val_bleu.update(bleu_score)
                    self.val_rouge_1.update(rouge_scores["rouge-1"]["f"])
                    self.val_rouge_2.update(rouge_scores["rouge-2"]["f"])
                    self.val_rouge_l.update(rouge_scores["rouge-l"]["f"])

                total_bleu.append(np.average(epoch_bleu))
                total_rouge_1.append(np.average(epoch_rouge_1))
                total_rouge_2.append(np.average(epoch_rouge_2))
                total_rouge_l.append(np.average(epoch_rouge_l))


                print('epochs: ' + str(epoch))
                print('average train loss: ' + str(self.train_loss.avg))
                print('average validation bleu score: ' + str(self.val_bleu.avg))
                print('average validation rouge-l score: ' + str(self.val_rouge_l.avg))
                # early stopping
                # Save model if bleu score is higher than the best
                if self.best_bleu < self.val_bleu.avg:
                    self.best_bleu = self.val_bleu.avg
                    checkpoint = {
                        'encoder': self.encoder,
                        'decoder': self.decoder,
                        'epoch': epoch
                    }
                    torch.save(checkpoint, self.log_path + '/Model_e%d_bleu%.3f.pt' % (epoch, self.val_bleu.avg))
                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.nepoch_no_imprv:
                        print("- early stopping {} epochs without " \
                              "improvement".format(nepoch_no_imprv))
                        break

        pandas_bleu = pd.DataFrame.from_dict({"bleu_validation": total_bleu})
        pandas_loss = pd.DataFrame.from_dict({"loss_train": total_loss})
        pandas_rouge_1 = pd.DataFrame.from_dict({"rouge_1_val": total_rouge_1})
        pandas_rouge_2 = pd.DataFrame.from_dict({"rouge_2_val": total_rouge_2})
        pandas_rouge_l = pd.DataFrame.from_dict({"rouge_l_val": total_rouge_l})

        pandas_loss.to_csv("./loss_train.csv", sep="\t", index=False)
        pandas_bleu.to_csv("./blue_validation.csv", sep="\t", index=False)
        pandas_rouge_1.to_csv("./rouge_1_validation.csv", sep="\t", index=False)
        pandas_rouge_2.to_csv("./rouge_2_validation.csv", sep="\t", index=False)
        pandas_rouge_l.to_csv("./rouge_l_validation.csv", sep="\t", index=False)


