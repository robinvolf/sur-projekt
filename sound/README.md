# Klasifikátor osoby pomocí zvuku

### Poznámky k datasetu
- Všechny vzory mají 1 kanál, vzorkovaný frekvencí 16kHz se vzorky, které mají 16bitové Inty

### Poznámky k funkcionalitě
1. Rozsekáme si příchozí signál na kratičké kousky, několik milisekund
2. Proženeme signál (teď už jen ten kousek) Hammingovým oknem
  - Utlumí to nespojitosti při pozdější Fourierově transformaci (ta předpokládá periodicitu signálu)
3. Proženeme filtrem na zesílení vysokých frekvencí
  - Člověk je na ně citlivý (pokud vzorkujeme např. 8kHz, tak nejvyšší frekvence budou 4kHz a na to jsou lidé opravdu citliví)
  - Není to *tak* důležité
4. Spočítám si Fourierovu transformaci signálu
  - Zajímá mě pouze modulová část spektra (ignoruju fázi, věříme, že na fázi člověk není citlivý)
  - Teď máme jakousi "charakteristiku", ale vůbec jsme si nepomohli, co se redukce dimenzí týče (pořád je to mnohadimenzionální vektor)
  - Pro FFT potřebujeme 2^N vzorků (napaddujeme to nulama)
  - Spektrum by mělo být symetrické, ale my si necháme jen kladnou část (je to redundantní)
5. Mám banku filtrů, na kterou se dívám jen jako na jakousi váhovací funkci, která vezme část spektra, naváhuje ji a sečte
  - Takové "trojúhelníčky", které se postupně zvětšují (na začátku úzké, na vyšších frekvencích široké)
  - Lidi mají logaritmické vnímaní frekvencí (posun o oktavů je vždy 2x vyšší, ale my všechny posuny vnímáme jako konstantní posuny) -- nemáme tam takové rozlišení, můžu podvzorkovat
  - Tímto dost snížím dimenzi
6. Signál zlogaritmujeme -- opět přiblížení se lidskému vnímání zvuku
  - "log mel banka filtrů"
  - Ale je to docela dost korelované -- sousední hodnoty jsou na sobě dost závislé (hodnota je pravděpodobně podobná svým sousedům)
  - Pomůže nám to v "gaussovštění" dat
7. PCA koeficientů
  - Kovarianční matice, vlastní vektory (směry variability), vlastní čísla (velikost variability)
  - Spočítám si skalární součin mezi datem a vlastními vektory -> nové souřadnice data (s nižší dimenzionalitou)
  - Aji nám to dekoreluje
  - U MFCC to nepromítáme do vlastních vektorů, ale do kosinové transformace
    - Otázka -- použít LDA? Přece klasifikujeme, mohlo by nám pomoct to promítnout do takového prostoru, kde se dají od sebe oddělit třídy!
