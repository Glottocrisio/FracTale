import numpy as np
from scipy.optimize import fsolve
from typing import List, Dict, Tuple
import re

class TextHierarchyDimension:
    def __init__(self):
        self.levels = ['sequence', 'section', 'passage', 'part', 'piece', 'segment']
    
    def split_into_segments(self, text: str) -> Dict[str, List[str]]:
        """Split text into hierarchical segments."""
        # Clean text
        text = text.strip()
        
        # Initialize hierarchy dictionary
        hierarchy = {}
        
        # Sequence is the full text
        hierarchy['sequence'] = [text]
        
        # Sections (split by multiple newlines and potential section markers)
        sections = re.split(r'\n\s*\n\s*\n+', text)
        hierarchy['section'] = [s.strip() for s in sections if s.strip()]
        
        # Passages (paragraphs)
        passages = re.split(r'\n\s*\n', text)
        hierarchy['passage'] = [p.strip() for p in passages if p.strip()]
        
        # Parts (split by single newline or strong punctuation)
        parts = re.split(r'\n|(?<=[.!?])\s+', text)
        hierarchy['part'] = [p.strip() for p in parts if p.strip()]
        
        # Pieces (sentences)
        pieces = re.split(r'[.!?]+', text)
        hierarchy['piece'] = [p.strip() for p in pieces if p.strip()]
        
        # Segments (phrases or clauses)
        segments = re.split(r'[,;:]|(?<=[.!?])\s+', text)
        hierarchy['segment'] = [s.strip() for s in segments if s.strip()]
        
        return hierarchy
    
    def calculate_sizes(self, hierarchy: Dict[str, List[str]]) -> Dict[str, int]:
        """Calculate size (in characters) for each level."""
        sizes = {}
        for level, texts in hierarchy.items():
            # Sum the lengths of all texts at this level
            sizes[level] = sum(len(text) for text in texts)
        return sizes
    
    def fractal_equation(self, d: float, size_ratios: List[float]) -> float:
        """
        The equation that needs to be solved:
        (size1/size2)^d + (size2/size3)^d + ... = 1
        """
        return sum(ratio**d for ratio in size_ratios) - 1
    
    def calculate_dimension(self, text: str) -> Tuple[float, Dict[str, int], List[float]]:
        """Calculate the fractal dimension of the text hierarchy."""
        # Get hierarchical splits
        hierarchy = self.split_into_segments(text)
        
        # Calculate sizes
        sizes = self.calculate_sizes(hierarchy)
        
        # Calculate size ratios
        ratios = []
        for i in range(len(self.levels) - 1):
            current_level = self.levels[i]
            next_level = self.levels[i + 1]
            ratio = sizes[current_level] / sizes[next_level]
            ratios.append(ratio)
        
        # Solve for dimension d
        # Initial guess for d
        d_guess = 1.0
        
        # Define the equation to solve
        equation = lambda d: self.fractal_equation(d, ratios)
        
        # Solve equation
        dimension = fsolve(equation, d_guess)[0]
        
        return dimension, sizes, ratios
    
    def analyze_text(self, text: str) -> Dict:
        """Perform complete analysis of text hierarchy."""
        dimension, sizes, ratios = self.calculate_dimension(text)
        
        # Calculate sum with found dimension to verify
        sum_with_d = sum(ratio**dimension for ratio in ratios)
        
        # Calculate individual terms
        terms = []
        for i in range(len(self.levels) - 1):
            current_level = self.levels[i]
            next_level = self.levels[i + 1]
            ratio = sizes[current_level] / sizes[next_level]
            term = ratio**dimension
            terms.append({
                'levels': f"{current_level}/{next_level}",
                'ratio': ratio,
                'term': term
            })
        
        return {
            'dimension': dimension,
            'sizes': sizes,
            'ratios': ratios,
            'terms': terms,
            'sum_verification': sum_with_d
        }

    def print_analysis(self, analysis: Dict):
        """Print detailed analysis results."""
        print(f"Fractal Dimension (d): {analysis['dimension']:.4f}")
        print("\nSizes at each level:")
        for level, size in analysis['sizes'].items():
            print(f"{level}: {size} characters")
        
        print("\nTerms of the equation:")
        for term in analysis['terms']:
            print(f"{term['levels']}: ratio = {term['ratio']:.4f}, term = {term['term']:.4f}")
        
        print(f"\nSum verification: {analysis['sum_verification']:.4f} (should be close to 1)")

# Example usage
if __name__ == "__main__":
    # Sample text with clear hierarchical structure
    sample_text = """Es war einmal ein M\u00fcller, der war arm, aber er hatte eine sch\u00f6ne Tochter. Nun traf es sich, da\u00df er mit dem K\u00f6nig zu sprechen kam, und um sich ein Ansehen zu geben, sagte er zu ihm: "Ich habe eine Tochter, die kann Stroh zu Gold spinnen." Der K\u00f6nig sprach zum M\u00fcller: "Das ist eine Kunst, die mir wohl gef\u00e4llt, wenn deine Tochter so geschickt ist, wie du sagst, so bring sie morgen in mein Schlo\u00df, da will ich sie auf die Probe stellen."
Als nun das M\u00e4dchen zu ihm gebracht ward, f\u00fchrte er es in eine Kammer, die ganz voll Stroh lag, gab ihr Rad und Haspel und sprach: "Jetzt mache dich an die Arbeit, und wenn du diese Nacht durch bis morgen fr\u00fch dieses Stroh nicht zu Gold versponnen hast, so mu\u00dft du sterben." Darauf schlo\u00df er die Kammer selbst zu, und sie blieb allein darin. Da sa\u00df nun die arme M\u00fcllerstochter und wu\u00dfte um ihr Leben keinen Rat: sie verstand gar nichts davon, wie man Stroh zu Gold spinnen konnte, und ihre Angst ward immer gr\u00f6\u00dfer, da\u00df sie endlich zu weinen anfing. Da ging auf einmal die T\u00fcre auf, und trat ein kleines M\u00e4nnchen herein und sprach: "Guten Abend, Jungfer M\u00fcllerin, warum weint Sie so sehr?""Ach," antwortete das M\u00e4dchen, "ich soll Stroh zu Gold spinnen und verstehe das nicht." Sprach das M\u00e4nnchen: "Was gibst du mir, wenn ich dirs spinne?" - "Mein Halsband," sagte das M\u00e4dchen. Das M\u00e4nnchen nahm das Halsband, setzte sich vor das R\u00e4dchen, und schnurr, schnurr, schnurr, dreimal gezogen, war die Spule voll. Dann steckte es eine andere auf, und schnurr, schnurr, schnurr, dreimal gezogen, war auch die zweite voll: und so gings fort bis zum Morgen, da war alles Stroh versponnen, und alle Spulen waren voll Gold.
Bei Sonnenaufgang kam schon der K\u00f6nig, und als er das Gold erblickte, erstaunte er und freute sich, aber sein Herz ward nur noch geldgieriger. Er lie\u00df die M\u00fcllerstochter in eine andere Kammer voll Stroh bringen, die noch viel gr\u00f6\u00dfer war, und befahl ihr, das auch in einer Nacht zu spinnen, wenn ihr das Leben lieb w\u00e4re. Das M\u00e4dchen wu\u00dfte sich nicht zu helfen und weinte, da ging abermals die T\u00fcre auf, und das kleine M\u00e4nnchen erschien und sprach: "Was gibst du mir, wenn ich dir das Stroh zu Gold spinne?""Meinen Ring von dem Finger," antwortete das M\u00e4dchen. Das M\u00e4nnchen nahm den Ring, fing wieder an zu schnurren mit dem Rade und hatte bis zum Morgen alles Stroh zu gl\u00e4nzendem Gold gesponnen. Der K\u00f6nig freute sich \u00fcber die Ma\u00dfen bei dem Anblick, war aber noch immer nicht Goldes satt, sondern lie\u00df die M\u00fcllerstochter in eine noch gr\u00f6\u00dfere Kammer voll Stroh bringen und sprach: "Die mu\u00dft du noch in dieser Nacht verspinnen: gelingt dir's aber, so sollst du meine Gemahlin werden." - "Wenn's auch eine M\u00fcllerstochter ist," dachte er, "eine reichere Frau finde ich in der ganzen Welt nicht." Als das M\u00e4dchen allein war, kam das M\u00e4nnlein zum drittenmal wieder und sprach: "Was gibst du mir, wenn ich dir noch diesmal das Stroh spinne?" - "Ich habe nichts mehr, das ich geben k\u00f6nnte," antwortete das M\u00e4dchen. "So versprich mir, wenn du K\u00f6nigin wirst, dein erstes Kind." - "Wer wei\u00df, wie das noch geht," dachte die M\u00fcllerstochter und wu\u00dfte sich auch in der Not nicht anders zu helfen; sie versprach also dem M\u00e4nnchen, was es verlangte, und das M\u00e4nnchen spann daf\u00fcr noch einmal das Stroh zu Gold. Und als am Morgen der K\u00f6nig kam und alles fand, wie er gew\u00fcnscht hatte, so hielt er Hochzeit mit ihr, und die sch\u00f6ne M\u00fcllerstochter ward eine K\u00f6nigin.
\u00dcber ein Jahr brachte sie ein sch\u00f6nes Kind zur Welt und dachte gar nicht mehr an das M\u00e4nnchen: da trat es pl\u00f6tzlich in ihre Kammer und sprach: "Nun gib mir, was du versprochen hast." Die K\u00f6nigin erschrak und bot dem M\u00e4nnchen alle Reicht\u00fcmer des K\u00f6nigreichs an, wenn es ihr das Kind lassen wollte: aber das M\u00e4nnchen sprach: "Nein, etwas Lebendes ist mir lieber als alle Sch\u00e4tze der Welt." Da fing die K\u00f6nigin so an zu jammern und zu weinen, da\u00df das M\u00e4nnchen Mitleiden mit ihr hatte: "Drei Tage will ich dir Zeit lassen," sprach er, "wenn du bis dahin meinen Namen wei\u00dft, so sollst du dein Kind behalten."
Nun besann sich die K\u00f6nigin die ganze Nacht \u00fcber auf alle Namen, die sie jemals geh\u00f6rt hatte, und schickte einen Boten \u00fcber Land, der sollte sich erkundigen weit und breit, was es sonst noch f\u00fcr Namen g\u00e4be. Als am andern Tag das M\u00e4nnchen kam, fing sie an mit Kaspar, Melchior, Balzer, und sagte alle Namen, die sie wu\u00dfte, nach der Reihe her, aber bei jedem sprach das M\u00e4nnlein: "So hei\u00df ich nicht." Den zweiten Tag lie\u00df sie in der Nachbarschaft herumfragen, wie die Leute da genannt w\u00fcrden, und sagte dem M\u00e4nnlein die ungew\u00f6hnlichsten und seltsamsten Namen vor "Hei\u00dft du vielleicht Rippenbiest oder Hammelswade oder Schn\u00fcrbein?" Aber es antwortete immer: "So hei\u00df ich nicht."
Den dritten Tag kam der Bote wieder zur\u00fcck und erz\u00e4hlte: "Neue Namen habe ich keinen einzigen finden k\u00f6nnen, aber wie ich an einen hohen Berg um die Waldecke kam, wo Fuchs und Has sich gute Nacht sagen, so sah ich da ein kleines Haus, und vor dem Haus brannte ein Feuer, und um das Feuer sprang ein gar zu l\u00e4cherliches M\u00e4nnchen, h\u00fcpfte auf einem Bein und schrie:"Heute back ich,Morgen brau ich,\u00dcbermorgen hol ich der K\u00f6nigin ihr Kind;Ach, wie gut ist, da\u00df niemand wei\u00df,da\u00df ich Rumpelstilzchen hei\u00df!"
Da k\u00f6nnt ihr denken, wie die K\u00f6nigin froh war, als sie den Namen h\u00f6rte, und als bald hernach das M\u00e4nnlein hereintrat und fragte: "Nun, Frau K\u00f6nigin, wie hei\u00df ich?" fragte sie erst: "Hei\u00dfest du Kunz?" - "Nein." - "Hei\u00dfest du Heinz?" - "Nein." - "Hei\u00dft du etwa Rumpelstilzchen?""Das hat dir der Teufel gesagt, das hat dir der Teufel gesagt," schrie das M\u00e4nnlein und stie\u00df mit dem rechten Fu\u00df vor Zorn so tief in die Erde, da\u00df es bis an den Leib hineinfuhr, dann packte es in seiner Wut den linken Fu\u00df mit beiden H\u00e4nden und ri\u00df sich selbst mitten entzwei.
"""
    analyzer = TextHierarchyDimension()
    analysis = analyzer.analyze_text(sample_text)
    analyzer.print_analysis(analysis)
