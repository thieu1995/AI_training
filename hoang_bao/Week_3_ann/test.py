from pyvi import ViTokenizer, ViPosTagger

ViTokenizer.tokenize(u"Trường đại học bách khoa hà nội")

print(ViPosTagger.postagging(ViTokenizer.tokenize(u"tôi thích hà nội vì nó thật là đẹp")))