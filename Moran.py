import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from nltk import sent_tokenize, word_tokenize
import csv
import os
from tabulate import tabulate


def moran_i(text, max_distance=20):
    clean_text = ''.join(char.lower() for char in text if char.isalpha())
    
    letter_counts = Counter(clean_text)
    total_letters = sum(letter_counts.values())
    
    try:
        mean_freq = total_letters / len(letter_counts)
    except ZeroDivisionError:
        pass
    
    positions = {char: [] for char in letter_counts}
    for i, char in enumerate(clean_text):
        positions[char].append(i)
    moran_values = []
    distances = range(1, max_distance + 1)
    
    for d in distances:
        numerator = 0
        denominator = 0
        
        for char, count in letter_counts.items():
            freq = count / total_letters
            diff = freq - mean_freq
            denominator += diff ** 2
            
            for i in range(len(positions[char])):
                for j in range(i + 1, len(positions[char])):
                    if abs(positions[char][i] - positions[char][j]) == d:
                        numerator += diff ** 2
        
        if denominator == 0:
            moran_i = 0
        else:
            moran_i = numerator / denominator
        
        moran_values.append(moran_i)
    
    return distances, moran_values

def plot_moran(distances, moran_values):
    plt.figure(figsize=(10, 6))
    plt.plot(distances, moran_values, 'b-')
    plt.xlabel('Distance')
    plt.ylabel("Moran's I")
    plt.title("Moran's I vs Distance")
    plt.grid(True)
    plt.show()



def split_into_clauses(text):
    sentences = sent_tokenize(text)
    clauses = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        clause = []
        for word in words:
            if word in [',', ';', ':', '(', ')', '\u2014', '\u2013', '-'] and clause:
                clauses.append(' '.join(clause))
                clause = []
            else:
                clause.append(word)
        if clause:
            clauses.append(' '.join(clause))
    return clauses

def moran_i_clauses(text, max_distance=20):
    clauses = split_into_clauses(text)
    
    clause_lengths = [len(clause) for clause in clauses]
    
    mean_length = np.mean(clause_lengths)
    
    moran_values = []
    distances = range(1, min(max_distance + 1, len(clauses)))
    
    for d in distances:
        numerator = 0
        denominator = 0
        
        for i in range(len(clauses)):
            diff_i = clause_lengths[i] - mean_length
            denominator += diff_i ** 2
            
            for j in range(i + 1, len(clauses)):
                if abs(i - j) == d:
                    diff_j = clause_lengths[j] - mean_length
                    numerator += diff_i * diff_j
        
        if denominator == 0:
            moran_i = 0
        else:
            moran_i = (len(clauses) / sum(range(1, len(clauses)))) * (numerator / denominator)
        
        moran_values.append(moran_i)
    
    return distances, moran_values

def plot_moran_clauses(distances, moran_values):
    plt.figure(figsize=(10, 6))
    plt.plot(distances, moran_values, 'b-')
    plt.xlabel('Distance (in clauses)')
    plt.ylabel("Moran's I")
    plt.title("Moran's I vs Distance for Clause Lengths")
    plt.grid(True)
    plt.show()


def plot_moran_sentences(distances, moran_values):
    plt.figure(figsize=(10, 6))
    plt.plot(distances, moran_values, 'b-')
    plt.xlabel('Distance (in sentences)')
    plt.ylabel("Moran's I")
    plt.title("Moran's I vs Distance for sentence Lengths")
    plt.grid(True)
    plt.show()

def moran_i_sentences(text, max_distance=20):
    sentences = sent_tokenize(text)
    
    sentence_lengths = [len(sentence) for sentence in sentences]
    
    mean_length = np.mean(sentence_lengths)
    
    moran_values = []
    distances = range(1, min(max_distance + 1, len(sentences)))
    
    for d in distances:
        numerator = 0
        denominator = 0
        
        for i in range(len(sentences)):
            diff_i = sentence_lengths[i] - mean_length
            denominator += diff_i ** 2
            
            for j in range(i + 1, len(sentences)):
                if abs(i - j) == d:
                    diff_j = sentence_lengths[j] - mean_length
                    numerator += diff_i * diff_j
        
        if denominator == 0:
            moran_i = 0
        else:
            moran_i = (len(sentences) / sum(range(1, len(sentences)))) * (numerator / denominator)
        
        moran_values.append(moran_i)
    
    return distances, moran_values


def calculate_fractal_dimension(distances, moran_values):
    try:
        distances = np.array(distances, dtype=float)
        moran_values = np.array(moran_values, dtype=float)
        
        if len(distances) < 2 or len(moran_values) < 2:
            return np.nan
            
        distances = distances[1:]
        moran_values = moran_values[1:]
        
        valid_mask = ~(np.isnan(distances) | np.isnan(moran_values) | 
                      (distances <= 0) | (moran_values == 0))
        
        valid_distances = distances[valid_mask]
        valid_moran = moran_values[valid_mask]
        
        if len(valid_distances) < 2:
            return np.nan
            
        log_distances = np.log(valid_distances)
        log_moran = np.log(np.abs(valid_moran))
        
        slope, _ = np.polyfit(log_distances, log_moran, 1)
        
        return 1 - slope / 2
        
    except Exception as e:
        print(f"Error in calculation: {str(e)}")
        return np.nan

def moran_i_episodes(text, max_distance=20):
    episodes = text.split('\n\n')
    
    episode_lengths = [len(episode.strip()) for episode in episodes]
    
    mean_length = np.mean(episode_lengths)
    
    moran_values = []
    distances = range(1, min(max_distance + 1, len(episodes)))
    
    for d in distances:
        numerator = 0
        denominator = 0
        
        for i in range(len(episodes)):
            diff_i = episode_lengths[i] - mean_length
            denominator += diff_i ** 2
            
            for j in range(i + 1, len(episodes)):
                if abs(i - j) == d:
                    diff_j = episode_lengths[j] - mean_length
                    numerator += diff_i * diff_j
        
        if denominator == 0:
            moran_i = 0
        else:
            moran_i = (len(episodes) / sum(range(1, len(episodes)))) * (numerator / denominator)
        
        moran_values.append(moran_i)
    
    return distances, moran_values

def plot_moran_words(distances, moran_values):
    plt.figure(figsize=(10, 6))
    plt.plot(distances, moran_values, 'b-')
    plt.xlabel('Distance (in words)')
    plt.ylabel("Moran's I")
    plt.title("Moran's I vs Distance for Word Lengths")
    plt.grid(True)
    plt.show()

def moran_i_words(text, max_distance=20):
    words = [word for word in word_tokenize(text) if word.isalnum()]
    
    word_lengths = [len(word) for word in words]
    
    mean_length = np.mean(word_lengths)
    
    moran_values = []
    distances = range(1, min(max_distance + 1, len(words)))
    
    for d in distances:
        numerator = 0
        denominator = 0
        
        for i in range(len(words)):
            diff_i = word_lengths[i] - mean_length
            denominator += diff_i ** 2
            
            for j in range(i + 1, len(words)):
                if abs(i - j) == d:
                    diff_j = word_lengths[j] - mean_length
                    numerator += diff_i * diff_j
        
        if denominator == 0:
            moran_i = 0
        else:
            moran_i = (len(words) / sum(range(1, len(words)))) * (numerator / denominator)
        
        moran_values.append(moran_i)
    
    return distances, moran_values

# # Example usage
# text = """
# "Die kl\u00e4ffenden Schn\u00fcrpelwesen tanzten durch die endlose Bl\u00e4tterwand, w\u00e4hrend gr\u00fcne Sp\u00e4tzelschwinger ihre Kr\u00e4uselmaschinen aufstellten."
# "Qu\u00e4k."
# "Der verdr\u00f6sselte Wurzelgeist versch\u00fcttete seine kostbaren Tr\u00f6pfelf\u00e4den \u00fcber den mondbeleuchteten Kn\u00f6delberg, bis die Br\u00f6selwesen anfingen zu quietschen und zu schnattern."
# "M\u00fcmpfelige Gr\u00fcnzelf\u00e4nger."
# "Die Schl\u00fcmpelbeeren explodierten in tausend winzige Fl\u00fcsterkr\u00e4mpfe."
# "Zwischen den kr\u00e4chzenden Bl\u00f6delschnecken und w\u00fcsten Kn\u00f6terichpflanzen erhob sich pl\u00f6tzlich ein gigantischer, leuchtender Tr\u00f6tenpilz, dessen Sp\u00e4tzelschwingen das gesamte Wurstengel\u00e4nde in ein magisches Gr\u00fcnzellicht tauchten."
# "Der Qu\u00e4tschenfrosch b\u00e4ngte."
# "Unter dem zerkn\u00fcllten Fl\u00fcsterhimmel sammelten sich die verschr\u00e4nkten Br\u00f6selgeister, um ihre j\u00e4hrliche Kn\u00f6delzeremonie abzuhalten."
# "Z\u00fcmp."
# "Die m\u00e4chtigen Schl\u00fcmpelwesen verschmolzen mit den dr\u00f6hnenden Tr\u00e4umelf\u00e4den zu einer einzigen, pulsierenden Masse aus Gr\u00fcbel und Gl\u00fcck."
# "Kr\u00e4chzende Bl\u00f6delschnecken verstreuten ihre leuchtenden Sp\u00e4tzelsporen \u00fcber das gesamte Land, w\u00e4hrend die Kn\u00f6ppelgeister in ihren Z\u00fcndelschr\u00e4nken kicherten und fl\u00fcsterten."
# "Knarz."
# """
# distances, moran_values = moran_i_clauses(text)
# plot_moran_clauses(distances, moran_values)

# # Calculate fractal dimension
# log_distances = np.log(distances[1:])  # Exclude distance 1
# log_moran = np.log(np.abs(moran_values[1:]))  # Use absolute values and exclude first
# slope, _ = np.polyfit(log_distances, log_moran, 1)
# fractal_dimension = 1 - slope / 2

# print(f"Estimated fractal dimension (clauses): {fractal_dimension:.4f}")
# # Example usage
# text = """
# Es war einmal ein M\u00fcller, der war arm, aber er hatte eine sch\u00f6ne Tochter. Nun traf es sich, da\u00df er mit dem K\u00f6nig zu sprechen kam, und um sich ein Ansehen zu geben, sagte er zu ihm: "Ich habe eine Tochter, die kann Stroh zu Gold spinnen." Der K\u00f6nig sprach zum M\u00fcller: "Das ist eine Kunst, die mir wohl gef\u00e4llt, wenn deine Tochter so geschickt ist, wie du sagst, so bring sie morgen in mein Schlo\u00df, da will ich sie auf die Probe stellen."
# Als nun das M\u00e4dchen zu ihm gebracht ward, f\u00fchrte er es in eine Kammer, die ganz voll Stroh lag, gab ihr Rad und Haspel und sprach: "Jetzt mache dich an die Arbeit, und wenn du diese Nacht durch bis morgen fr\u00fch dieses Stroh nicht zu Gold versponnen hast, so mu\u00dft du sterben." Darauf schlo\u00df er die Kammer selbst zu, und sie blieb allein darin. Da sa\u00df nun die arme M\u00fcllerstochter und wu\u00dfte um ihr Leben keinen Rat: sie verstand gar nichts davon, wie man Stroh zu Gold spinnen konnte, und ihre Angst ward immer gr\u00f6\u00dfer, da\u00df sie endlich zu weinen anfing. Da ging auf einmal die T\u00fcre auf, und trat ein kleines M\u00e4nnchen herein und sprach: "Guten Abend, Jungfer M\u00fcllerin, warum weint Sie so sehr?""Ach," antwortete das M\u00e4dchen, "ich soll Stroh zu Gold spinnen und verstehe das nicht." Sprach das M\u00e4nnchen: "Was gibst du mir, wenn ich dirs spinne?" - "Mein Halsband," sagte das M\u00e4dchen. Das M\u00e4nnchen nahm das Halsband, setzte sich vor das R\u00e4dchen, und schnurr, schnurr, schnurr, dreimal gezogen, war die Spule voll. Dann steckte es eine andere auf, und schnurr, schnurr, schnurr, dreimal gezogen, war auch die zweite voll: und so gings fort bis zum Morgen, da war alles Stroh versponnen, und alle Spulen waren voll Gold.
# Bei Sonnenaufgang kam schon der K\u00f6nig, und als er das Gold erblickte, erstaunte er und freute sich, aber sein Herz ward nur noch geldgieriger. Er lie\u00df die M\u00fcllerstochter in eine andere Kammer voll Stroh bringen, die noch viel gr\u00f6\u00dfer war, und befahl ihr, das auch in einer Nacht zu spinnen, wenn ihr das Leben lieb w\u00e4re. Das M\u00e4dchen wu\u00dfte sich nicht zu helfen und weinte, da ging abermals die T\u00fcre auf, und das kleine M\u00e4nnchen erschien und sprach: "Was gibst du mir, wenn ich dir das Stroh zu Gold spinne?""Meinen Ring von dem Finger," antwortete das M\u00e4dchen. Das M\u00e4nnchen nahm den Ring, fing wieder an zu schnurren mit dem Rade und hatte bis zum Morgen alles Stroh zu gl\u00e4nzendem Gold gesponnen. Der K\u00f6nig freute sich \u00fcber die Ma\u00dfen bei dem Anblick, war aber noch immer nicht Goldes satt, sondern lie\u00df die M\u00fcllerstochter in eine noch gr\u00f6\u00dfere Kammer voll Stroh bringen und sprach: "Die mu\u00dft du noch in dieser Nacht verspinnen: gelingt dir's aber, so sollst du meine Gemahlin werden." - "Wenn's auch eine M\u00fcllerstochter ist," dachte er, "eine reichere Frau finde ich in der ganzen Welt nicht." Als das M\u00e4dchen allein war, kam das M\u00e4nnlein zum drittenmal wieder und sprach: "Was gibst du mir, wenn ich dir noch diesmal das Stroh spinne?" - "Ich habe nichts mehr, das ich geben k\u00f6nnte," antwortete das M\u00e4dchen. "So versprich mir, wenn du K\u00f6nigin wirst, dein erstes Kind." - "Wer wei\u00df, wie das noch geht," dachte die M\u00fcllerstochter und wu\u00dfte sich auch in der Not nicht anders zu helfen; sie versprach also dem M\u00e4nnchen, was es verlangte, und das M\u00e4nnchen spann daf\u00fcr noch einmal das Stroh zu Gold. Und als am Morgen der K\u00f6nig kam und alles fand, wie er gew\u00fcnscht hatte, so hielt er Hochzeit mit ihr, und die sch\u00f6ne M\u00fcllerstochter ward eine K\u00f6nigin.
# \u00dcber ein Jahr brachte sie ein sch\u00f6nes Kind zur Welt und dachte gar nicht mehr an das M\u00e4nnchen: da trat es pl\u00f6tzlich in ihre Kammer und sprach: "Nun gib mir, was du versprochen hast." Die K\u00f6nigin erschrak und bot dem M\u00e4nnchen alle Reicht\u00fcmer des K\u00f6nigreichs an, wenn es ihr das Kind lassen wollte: aber das M\u00e4nnchen sprach: "Nein, etwas Lebendes ist mir lieber als alle Sch\u00e4tze der Welt." Da fing die K\u00f6nigin so an zu jammern und zu weinen, da\u00df das M\u00e4nnchen Mitleiden mit ihr hatte: "Drei Tage will ich dir Zeit lassen," sprach er, "wenn du bis dahin meinen Namen wei\u00dft, so sollst du dein Kind behalten."
# Nun besann sich die K\u00f6nigin die ganze Nacht \u00fcber auf alle Namen, die sie jemals geh\u00f6rt hatte, und schickte einen Boten \u00fcber Land, der sollte sich erkundigen weit und breit, was es sonst noch f\u00fcr Namen g\u00e4be. Als am andern Tag das M\u00e4nnchen kam, fing sie an mit Kaspar, Melchior, Balzer, und sagte alle Namen, die sie wu\u00dfte, nach der Reihe her, aber bei jedem sprach das M\u00e4nnlein: "So hei\u00df ich nicht." Den zweiten Tag lie\u00df sie in der Nachbarschaft herumfragen, wie die Leute da genannt w\u00fcrden, und sagte dem M\u00e4nnlein die ungew\u00f6hnlichsten und seltsamsten Namen vor "Hei\u00dft du vielleicht Rippenbiest oder Hammelswade oder Schn\u00fcrbein?" Aber es antwortete immer: "So hei\u00df ich nicht."
# Den dritten Tag kam der Bote wieder zur\u00fcck und erz\u00e4hlte: "Neue Namen habe ich keinen einzigen finden k\u00f6nnen, aber wie ich an einen hohen Berg um die Waldecke kam, wo Fuchs und Has sich gute Nacht sagen, so sah ich da ein kleines Haus, und vor dem Haus brannte ein Feuer, und um das Feuer sprang ein gar zu l\u00e4cherliches M\u00e4nnchen, h\u00fcpfte auf einem Bein und schrie:"Heute back ich,Morgen brau ich,\u00dcbermorgen hol ich der K\u00f6nigin ihr Kind;Ach, wie gut ist, da\u00df niemand wei\u00df,da\u00df ich Rumpelstilzchen hei\u00df!"
# Da k\u00f6nnt ihr denken, wie die K\u00f6nigin froh war, als sie den Namen h\u00f6rte, und als bald hernach das M\u00e4nnlein hereintrat und fragte: "Nun, Frau K\u00f6nigin, wie hei\u00df ich?" fragte sie erst: "Hei\u00dfest du Kunz?" - "Nein." - "Hei\u00dfest du Heinz?" - "Nein." - "Hei\u00dft du etwa Rumpelstilzchen?""Das hat dir der Teufel gesagt, das hat dir der Teufel gesagt," schrie das M\u00e4nnlein und stie\u00df mit dem rechten Fu\u00df vor Zorn so tief in die Erde, da\u00df es bis an den Leib hineinfuhr, dann packte es in seiner Wut den linken Fu\u00df mit beiden H\u00e4nden und ri\u00df sich selbst mitten entzwei.
# """

# # distances, moran_values = moran_i(text)
# # #plot_moran(distances, moran_values)

# # # Calculate fractal dimension
# # log_distances = np.log(distances[1:])  # Exclude distance 1
# # log_moran = np.log(moran_values[1:])
# # slope, _ = np.polyfit(log_distances, log_moran, 1)
# # fractal_dimension = 1 - slope / 2

# # print(f"Estimated fractal dimension (letters): {fractal_dimension:.4f}")




# # Example usage for sentences
# text = """
# Es war einmal ein M\u00fcller, der war arm, aber er hatte eine sch\u00f6ne Tochter. Nun traf es sich, da\u00df er mit dem K\u00f6nig zu sprechen kam, und um sich ein Ansehen zu geben, sagte er zu ihm: "Ich habe eine Tochter, die kann Stroh zu Gold spinnen." Der K\u00f6nig sprach zum M\u00fcller: "Das ist eine Kunst, die mir wohl gef\u00e4llt, wenn deine Tochter so geschickt ist, wie du sagst, so bring sie morgen in mein Schlo\u00df, da will ich sie auf die Probe stellen."
# Als nun das M\u00e4dchen zu ihm gebracht ward, f\u00fchrte er es in eine Kammer, die ganz voll Stroh lag, gab ihr Rad und Haspel und sprach: "Jetzt mache dich an die Arbeit, und wenn du diese Nacht durch bis morgen fr\u00fch dieses Stroh nicht zu Gold versponnen hast, so mu\u00dft du sterben." Darauf schlo\u00df er die Kammer selbst zu, und sie blieb allein darin. Da sa\u00df nun die arme M\u00fcllerstochter und wu\u00dfte um ihr Leben keinen Rat: sie verstand gar nichts davon, wie man Stroh zu Gold spinnen konnte, und ihre Angst ward immer gr\u00f6\u00dfer, da\u00df sie endlich zu weinen anfing. Da ging auf einmal die T\u00fcre auf, und trat ein kleines M\u00e4nnchen herein und sprach: "Guten Abend, Jungfer M\u00fcllerin, warum weint Sie so sehr?""Ach," antwortete das M\u00e4dchen, "ich soll Stroh zu Gold spinnen und verstehe das nicht." Sprach das M\u00e4nnchen: "Was gibst du mir, wenn ich dirs spinne?" - "Mein Halsband," sagte das M\u00e4dchen. Das M\u00e4nnchen nahm das Halsband, setzte sich vor das R\u00e4dchen, und schnurr, schnurr, schnurr, dreimal gezogen, war die Spule voll. Dann steckte es eine andere auf, und schnurr, schnurr, schnurr, dreimal gezogen, war auch die zweite voll: und so gings fort bis zum Morgen, da war alles Stroh versponnen, und alle Spulen waren voll Gold.
# Bei Sonnenaufgang kam schon der K\u00f6nig, und als er das Gold erblickte, erstaunte er und freute sich, aber sein Herz ward nur noch geldgieriger. Er lie\u00df die M\u00fcllerstochter in eine andere Kammer voll Stroh bringen, die noch viel gr\u00f6\u00dfer war, und befahl ihr, das auch in einer Nacht zu spinnen, wenn ihr das Leben lieb w\u00e4re. Das M\u00e4dchen wu\u00dfte sich nicht zu helfen und weinte, da ging abermals die T\u00fcre auf, und das kleine M\u00e4nnchen erschien und sprach: "Was gibst du mir, wenn ich dir das Stroh zu Gold spinne?""Meinen Ring von dem Finger," antwortete das M\u00e4dchen. Das M\u00e4nnchen nahm den Ring, fing wieder an zu schnurren mit dem Rade und hatte bis zum Morgen alles Stroh zu gl\u00e4nzendem Gold gesponnen. Der K\u00f6nig freute sich \u00fcber die Ma\u00dfen bei dem Anblick, war aber noch immer nicht Goldes satt, sondern lie\u00df die M\u00fcllerstochter in eine noch gr\u00f6\u00dfere Kammer voll Stroh bringen und sprach: "Die mu\u00dft du noch in dieser Nacht verspinnen: gelingt dir's aber, so sollst du meine Gemahlin werden." - "Wenn's auch eine M\u00fcllerstochter ist," dachte er, "eine reichere Frau finde ich in der ganzen Welt nicht." Als das M\u00e4dchen allein war, kam das M\u00e4nnlein zum drittenmal wieder und sprach: "Was gibst du mir, wenn ich dir noch diesmal das Stroh spinne?" - "Ich habe nichts mehr, das ich geben k\u00f6nnte," antwortete das M\u00e4dchen. "So versprich mir, wenn du K\u00f6nigin wirst, dein erstes Kind." - "Wer wei\u00df, wie das noch geht," dachte die M\u00fcllerstochter und wu\u00dfte sich auch in der Not nicht anders zu helfen; sie versprach also dem M\u00e4nnchen, was es verlangte, und das M\u00e4nnchen spann daf\u00fcr noch einmal das Stroh zu Gold. Und als am Morgen der K\u00f6nig kam und alles fand, wie er gew\u00fcnscht hatte, so hielt er Hochzeit mit ihr, und die sch\u00f6ne M\u00fcllerstochter ward eine K\u00f6nigin.
# \u00dcber ein Jahr brachte sie ein sch\u00f6nes Kind zur Welt und dachte gar nicht mehr an das M\u00e4nnchen: da trat es pl\u00f6tzlich in ihre Kammer und sprach: "Nun gib mir, was du versprochen hast." Die K\u00f6nigin erschrak und bot dem M\u00e4nnchen alle Reicht\u00fcmer des K\u00f6nigreichs an, wenn es ihr das Kind lassen wollte: aber das M\u00e4nnchen sprach: "Nein, etwas Lebendes ist mir lieber als alle Sch\u00e4tze der Welt." Da fing die K\u00f6nigin so an zu jammern und zu weinen, da\u00df das M\u00e4nnchen Mitleiden mit ihr hatte: "Drei Tage will ich dir Zeit lassen," sprach er, "wenn du bis dahin meinen Namen wei\u00dft, so sollst du dein Kind behalten."
# Nun besann sich die K\u00f6nigin die ganze Nacht \u00fcber auf alle Namen, die sie jemals geh\u00f6rt hatte, und schickte einen Boten \u00fcber Land, der sollte sich erkundigen weit und breit, was es sonst noch f\u00fcr Namen g\u00e4be. Als am andern Tag das M\u00e4nnchen kam, fing sie an mit Kaspar, Melchior, Balzer, und sagte alle Namen, die sie wu\u00dfte, nach der Reihe her, aber bei jedem sprach das M\u00e4nnlein: "So hei\u00df ich nicht." Den zweiten Tag lie\u00df sie in der Nachbarschaft herumfragen, wie die Leute da genannt w\u00fcrden, und sagte dem M\u00e4nnlein die ungew\u00f6hnlichsten und seltsamsten Namen vor "Hei\u00dft du vielleicht Rippenbiest oder Hammelswade oder Schn\u00fcrbein?" Aber es antwortete immer: "So hei\u00df ich nicht."
# Den dritten Tag kam der Bote wieder zur\u00fcck und erz\u00e4hlte: "Neue Namen habe ich keinen einzigen finden k\u00f6nnen, aber wie ich an einen hohen Berg um die Waldecke kam, wo Fuchs und Has sich gute Nacht sagen, so sah ich da ein kleines Haus, und vor dem Haus brannte ein Feuer, und um das Feuer sprang ein gar zu l\u00e4cherliches M\u00e4nnchen, h\u00fcpfte auf einem Bein und schrie:"Heute back ich,Morgen brau ich,\u00dcbermorgen hol ich der K\u00f6nigin ihr Kind;Ach, wie gut ist, da\u00df niemand wei\u00df,da\u00df ich Rumpelstilzchen hei\u00df!"
# Da k\u00f6nnt ihr denken, wie die K\u00f6nigin froh war, als sie den Namen h\u00f6rte, und als bald hernach das M\u00e4nnlein hereintrat und fragte: "Nun, Frau K\u00f6nigin, wie hei\u00df ich?" fragte sie erst: "Hei\u00dfest du Kunz?" - "Nein." - "Hei\u00dfest du Heinz?" - "Nein." - "Hei\u00dft du etwa Rumpelstilzchen?""Das hat dir der Teufel gesagt, das hat dir der Teufel gesagt," schrie das M\u00e4nnlein und stie\u00df mit dem rechten Fu\u00df vor Zorn so tief in die Erde, da\u00df es bis an den Leib hineinfuhr, dann packte es in seiner Wut den linken Fu\u00df mit beiden H\u00e4nden und ri\u00df sich selbst mitten entzwei.

# """

# # distances_sent, moran_values_sent = moran_i_sentences(text)
# # #plot_moran_sentences(distances_sent, moran_values_sent, "Moran's I vs Distance for Sentence Lengths")
# # fractal_dimension_sent = calculate_fractal_dimension(distances_sent, moran_values_sent)
# # print(f"Estimated fractal dimension (sentences): {fractal_dimension_sent:.4f}")


# text_with_episodes = """
# Es war einmal ein M\u00fcller, der war arm, aber er hatte eine sch\u00f6ne Tochter. Nun traf es sich, da\u00df er mit dem K\u00f6nig zu sprechen kam, und um sich ein Ansehen zu geben, sagte er zu ihm: "Ich habe eine Tochter, die kann Stroh zu Gold spinnen." Der K\u00f6nig sprach zum M\u00fcller: "Das ist eine Kunst, die mir wohl gef\u00e4llt, wenn deine Tochter so geschickt ist, wie du sagst, so bring sie morgen in mein Schlo\u00df, da will ich sie auf die Probe stellen."
# Als nun das M\u00e4dchen zu ihm gebracht ward, f\u00fchrte er es in eine Kammer, die ganz voll Stroh lag, gab ihr Rad und Haspel und sprach: "Jetzt mache dich an die Arbeit, und wenn du diese Nacht durch bis morgen fr\u00fch dieses Stroh nicht zu Gold versponnen hast, so mu\u00dft du sterben." Darauf schlo\u00df er die Kammer selbst zu, und sie blieb allein darin. Da sa\u00df nun die arme M\u00fcllerstochter und wu\u00dfte um ihr Leben keinen Rat: sie verstand gar nichts davon, wie man Stroh zu Gold spinnen konnte, und ihre Angst ward immer gr\u00f6\u00dfer, da\u00df sie endlich zu weinen anfing. Da ging auf einmal die T\u00fcre auf, und trat ein kleines M\u00e4nnchen herein und sprach: "Guten Abend, Jungfer M\u00fcllerin, warum weint Sie so sehr?""Ach," antwortete das M\u00e4dchen, "ich soll Stroh zu Gold spinnen und verstehe das nicht." Sprach das M\u00e4nnchen: "Was gibst du mir, wenn ich dirs spinne?" - "Mein Halsband," sagte das M\u00e4dchen. Das M\u00e4nnchen nahm das Halsband, setzte sich vor das R\u00e4dchen, und schnurr, schnurr, schnurr, dreimal gezogen, war die Spule voll. Dann steckte es eine andere auf, und schnurr, schnurr, schnurr, dreimal gezogen, war auch die zweite voll: und so gings fort bis zum Morgen, da war alles Stroh versponnen, und alle Spulen waren voll Gold.
# Bei Sonnenaufgang kam schon der K\u00f6nig, und als er das Gold erblickte, erstaunte er und freute sich, aber sein Herz ward nur noch geldgieriger. Er lie\u00df die M\u00fcllerstochter in eine andere Kammer voll Stroh bringen, die noch viel gr\u00f6\u00dfer war, und befahl ihr, das auch in einer Nacht zu spinnen, wenn ihr das Leben lieb w\u00e4re. Das M\u00e4dchen wu\u00dfte sich nicht zu helfen und weinte, da ging abermals die T\u00fcre auf, und das kleine M\u00e4nnchen erschien und sprach: "Was gibst du mir, wenn ich dir das Stroh zu Gold spinne?""Meinen Ring von dem Finger," antwortete das M\u00e4dchen. Das M\u00e4nnchen nahm den Ring, fing wieder an zu schnurren mit dem Rade und hatte bis zum Morgen alles Stroh zu gl\u00e4nzendem Gold gesponnen. Der K\u00f6nig freute sich \u00fcber die Ma\u00dfen bei dem Anblick, war aber noch immer nicht Goldes satt, sondern lie\u00df die M\u00fcllerstochter in eine noch gr\u00f6\u00dfere Kammer voll Stroh bringen und sprach: "Die mu\u00dft du noch in dieser Nacht verspinnen: gelingt dir's aber, so sollst du meine Gemahlin werden." - "Wenn's auch eine M\u00fcllerstochter ist," dachte er, "eine reichere Frau finde ich in der ganzen Welt nicht." Als das M\u00e4dchen allein war, kam das M\u00e4nnlein zum drittenmal wieder und sprach: "Was gibst du mir, wenn ich dir noch diesmal das Stroh spinne?" - "Ich habe nichts mehr, das ich geben k\u00f6nnte," antwortete das M\u00e4dchen. "So versprich mir, wenn du K\u00f6nigin wirst, dein erstes Kind." - "Wer wei\u00df, wie das noch geht," dachte die M\u00fcllerstochter und wu\u00dfte sich auch in der Not nicht anders zu helfen; sie versprach also dem M\u00e4nnchen, was es verlangte, und das M\u00e4nnchen spann daf\u00fcr noch einmal das Stroh zu Gold. Und als am Morgen der K\u00f6nig kam und alles fand, wie er gew\u00fcnscht hatte, so hielt er Hochzeit mit ihr, und die sch\u00f6ne M\u00fcllerstochter ward eine K\u00f6nigin.
# \u00dcber ein Jahr brachte sie ein sch\u00f6nes Kind zur Welt und dachte gar nicht mehr an das M\u00e4nnchen: da trat es pl\u00f6tzlich in ihre Kammer und sprach: "Nun gib mir, was du versprochen hast." Die K\u00f6nigin erschrak und bot dem M\u00e4nnchen alle Reicht\u00fcmer des K\u00f6nigreichs an, wenn es ihr das Kind lassen wollte: aber das M\u00e4nnchen sprach: "Nein, etwas Lebendes ist mir lieber als alle Sch\u00e4tze der Welt." Da fing die K\u00f6nigin so an zu jammern und zu weinen, da\u00df das M\u00e4nnchen Mitleiden mit ihr hatte: "Drei Tage will ich dir Zeit lassen," sprach er, "wenn du bis dahin meinen Namen wei\u00dft, so sollst du dein Kind behalten."
# Nun besann sich die K\u00f6nigin die ganze Nacht \u00fcber auf alle Namen, die sie jemals geh\u00f6rt hatte, und schickte einen Boten \u00fcber Land, der sollte sich erkundigen weit und breit, was es sonst noch f\u00fcr Namen g\u00e4be. Als am andern Tag das M\u00e4nnchen kam, fing sie an mit Kaspar, Melchior, Balzer, und sagte alle Namen, die sie wu\u00dfte, nach der Reihe her, aber bei jedem sprach das M\u00e4nnlein: "So hei\u00df ich nicht." Den zweiten Tag lie\u00df sie in der Nachbarschaft herumfragen, wie die Leute da genannt w\u00fcrden, und sagte dem M\u00e4nnlein die ungew\u00f6hnlichsten und seltsamsten Namen vor "Hei\u00dft du vielleicht Rippenbiest oder Hammelswade oder Schn\u00fcrbein?" Aber es antwortete immer: "So hei\u00df ich nicht."
# Den dritten Tag kam der Bote wieder zur\u00fcck und erz\u00e4hlte: "Neue Namen habe ich keinen einzigen finden k\u00f6nnen, aber wie ich an einen hohen Berg um die Waldecke kam, wo Fuchs und Has sich gute Nacht sagen, so sah ich da ein kleines Haus, und vor dem Haus brannte ein Feuer, und um das Feuer sprang ein gar zu l\u00e4cherliches M\u00e4nnchen, h\u00fcpfte auf einem Bein und schrie:"Heute back ich,Morgen brau ich,\u00dcbermorgen hol ich der K\u00f6nigin ihr Kind;Ach, wie gut ist, da\u00df niemand wei\u00df,da\u00df ich Rumpelstilzchen hei\u00df!"
# Da k\u00f6nnt ihr denken, wie die K\u00f6nigin froh war, als sie den Namen h\u00f6rte, und als bald hernach das M\u00e4nnlein hereintrat und fragte: "Nun, Frau K\u00f6nigin, wie hei\u00df ich?" fragte sie erst: "Hei\u00dfest du Kunz?" - "Nein." - "Hei\u00dfest du Heinz?" - "Nein." - "Hei\u00dft du etwa Rumpelstilzchen?""Das hat dir der Teufel gesagt, das hat dir der Teufel gesagt," schrie das M\u00e4nnlein und stie\u00df mit dem rechten Fu\u00df vor Zorn so tief in die Erde, da\u00df es bis an den Leib hineinfuhr, dann packte es in seiner Wut den linken Fu\u00df mit beiden H\u00e4nden und ri\u00df sich selbst mitten entzwei.

# """

# # distances_ep, moran_values_ep = moran_i_episodes(text_with_episodes)
# # #plot_moran(distances_ep, moran_values_ep, "Moran's I vs Distance for Episode Lengths")
# # fractal_dimension_ep = calculate_fractal_dimension(distances_ep, moran_values_ep)
# # print(f"Estimated fractal dimension (episodes): {fractal_dimension_ep:.4f}")

# # #Create a loop here to perform this analysis on every corpus for every language. 
# #Their result will be stored in tables and put in the paper's appendix

# corpora = ['europeana_stories_de.txt']

# for corpus in corpora:
#     csv_filename = f"{corpus}_results.csv".replace('.txt', '')
#     latex_filename = f"{corpus}_results.tex".replace('.txt', '')
    
#     with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
#         csvwriter = csv.writer(csvfile)
#         csvwriter.writerow(['Story', 'Fractal Dimension (Letters)', 'Fractal Dimension (Words)', 'Fractal Dimension (Clauses)', 'Fractal Dimension (Sentences)'])
    
#     latex_tables = []
    
#     with open(corpus, 'r', encoding='iso-8859-1') as file:
#         content = file.read()
#     tales = content.split('---\n\n')
    
#     for i, tale in enumerate(tales, 1):
#         tale_results = [f"Story {i}"]
        
#         # Letters 
#         distances, moran_values = moran_i(tale)
#         #plot_moran(distances, moran_values)
#         fractal_dimension_letters = calculate_fractal_dimension(distances, moran_values)
#         print(f"Estimated fractal dimension (letters): {fractal_dimension_letters:.4f}")
#         tale_results.append(f"{fractal_dimension_letters:.4f}")
        
#         # Words 
#         distances_words, moran_values_words = moran_i_words(tale)
#         #plot_moran_clauses(distances_clauses, moran_values_clauses)
#         fractal_dimension_words = calculate_fractal_dimension(distances_words, moran_values_words)
#         print(f"Estimated fractal dimension (words): {fractal_dimension_words:.4f}")
#         tale_results.append(f"{fractal_dimension_words:.4f}")

#         # Clauses 
#         distances_clauses, moran_values_clauses = moran_i_clauses(tale)
#         #plot_moran_clauses(distances_clauses, moran_values_clauses)
#         fractal_dimension_clauses = calculate_fractal_dimension(distances_clauses, moran_values_clauses)
#         print(f"Estimated fractal dimension (clauses): {fractal_dimension_clauses:.4f}")
#         tale_results.append(f"{fractal_dimension_clauses:.4f}")
        
#         # Sentences 
#         distances_sent, moran_values_sent = moran_i_sentences(tale)
#         #plot_moran_sentences(distances_sent, moran_values_sent)
#         fractal_dimension_sent = calculate_fractal_dimension(distances_sent, moran_values_sent)
#         print(f"Estimated fractal dimension (sentences): {fractal_dimension_sent:.4f}")
#         tale_results.append(f"{fractal_dimension_sent:.4f}")
        
        
#         # Append results to CSV
#         with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
#             csvwriter = csv.writer(csvfile)
#             csvwriter.writerow(tale_results)

#         if i == 30:
#             break
        
#     # Create LaTeX table
#     with open(csv_filename, 'r', encoding='utf-8') as csvfile:
#         csvreader = csv.reader(csvfile)
#         table_data = list(csvreader)
    
#     latex_table = tabulate(table_data[1:], headers=table_data[0], tablefmt="latex_booktabs")
    
#     # Write LaTeX table to file
#     with open(latex_filename, 'w', encoding='utf-8') as latexfile:
#         latexfile.write("\\begin{table}[h]\n\\centering\n")
#         latexfile.write(f"\\caption{{Fractal Dimensions for {corpus}}}\n")
#         latexfile.write("\\label{tab:fractal-dimensions-" + corpus + "}\n")
#         latexfile.write(latex_table)
#         latexfile.write("\n\\end{table}")
    
#     print(f"Results for {corpus} exported to {csv_filename} and {latex_filename}")

#     print("All analyses completed and results exported.")


corpora = ['europeana_stories_en.txt'] #'grimm_tales_de.txt', 'ugly_de_grimm_tales.txt', 'ugly_en_grimm_tales.txt', 'ugly_es_grimm_tales.txt', 'ugly_it_grimm_tales.txt', 'europeana_stories_de.txt','grimm_tales_es.txt','grimm_tales_it.txt'

for corpus in corpora:
    #csv_filename = f"{corpus}_results.csv".replace('.txt', '')
    #latex_filename = f"{corpus}_results.tex".replace('.txt', '')
    
    csv_filename = f"{corpus}_results.csv".replace('.txt', '')
    latex_filename = f"{corpus}_results.tex".replace('.txt', '')

    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Tale', 'Fractal Dimension (Letters)', 'Fractal Dimension (Words)', 'Fractal Dimension (Clauses)', 'Fractal Dimension (Sentences)', 'Fractal Dimension (Episodes)'])
    
    latex_tables = []
    
    with open(corpus, 'r', encoding='iso-8859-1') as file:
        content = file.read()
    tales = content.split('--------------------------------------------------\n\n')
    
    for i, tale in enumerate(tales, 1):
        #if i < 14:
        #    continue
        tale_results = [f"Tale {i}"]
        
        # Letters 
        distances, moran_values = moran_i(tale)
        #plot_moran(distances, moran_values)
        fractal_dimension_letters = calculate_fractal_dimension(distances, moran_values)
        print(f"Estimated fractal dimension (letters): {fractal_dimension_letters:.4f}")
        tale_results.append(f"{fractal_dimension_letters:.4f}")
        
        # Words 
        distances_words, moran_values_words = moran_i_words(tale)
        #plot_moran_clauses(distances_clauses, moran_values_clauses)
        fractal_dimension_words = calculate_fractal_dimension(distances_words, moran_values_words)
        print(f"Estimated fractal dimension (words): {fractal_dimension_words:.4f}")
        tale_results.append(f"{fractal_dimension_words:.4f}")

        # Clauses 
        distances_clauses, moran_values_clauses = moran_i_clauses(tale)
        #plot_moran_clauses(distances_clauses, moran_values_clauses)
        fractal_dimension_clauses = calculate_fractal_dimension(distances_clauses, moran_values_clauses)
        print(f"Estimated fractal dimension (clauses): {fractal_dimension_clauses:.4f}")
        tale_results.append(f"{fractal_dimension_clauses:.4f}")
        
        # Sentences 
        distances_sent, moran_values_sent = moran_i_sentences(tale)
        #plot_moran_sentences(distances_sent, moran_values_sent)
        fractal_dimension_sent = calculate_fractal_dimension(distances_sent, moran_values_sent)
        print(f"Estimated fractal dimension (sentences): {fractal_dimension_sent:.4f}")
        tale_results.append(f"{fractal_dimension_sent:.4f}")
        
        # Episodes 
        distances_ep, moran_values_ep = moran_i_episodes(tale)
        #plot_moran_sentences(distances_sent, moran_values_sent)
        fractal_dimension_sent = calculate_fractal_dimension(distances_ep, moran_values_ep)
        print(f"Estimated fractal dimension (episodes): {fractal_dimension_sent:.4f}")
        tale_results.append(f"{fractal_dimension_sent:.4f}")
        
        with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(tale_results)

        if i ==6:
            break
        
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        table_data = list(csvreader)
    
    latex_table = tabulate(table_data[1:], headers=table_data[0], tablefmt="latex_booktabs")
    
    with open(latex_filename, 'w', encoding='utf-8') as latexfile:
        latexfile.write("\\begin{table}[h]\n\\centering\n")
        latexfile.write(f"\\caption{{Fractal Dimensions for {corpus}}}\n")
        latexfile.write("\\label{tab:fractal-dimensions-" + corpus + "}\n")
        latexfile.write(latex_table)
        latexfile.write("\n\\end{table}")
    
    print(f"Results for {corpus} exported to {csv_filename} and {latex_filename}")

    print("All analyses completed and results exported.")