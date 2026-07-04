"""
Evaluacijski skup pitanja za RAGAS evaluaciju RAG pipeline-a.

Pokriva svih 5 domenskih dokumenata:
  A) Zakon o OIE i visokoučinkovitoj kogeneraciji         (4 pitanja)
  B) Priručnik o postupcima ishođenja dozvola              (4 pitanja)
  C) Metodologija energetskog pregleda zgrada 2021         (4 pitanja)
  D) Tehnički propis o sustavima ventilacije               (4 pitanja)
  E) Javni poziv EnU-6/25 (fotonaponske elektrane)         (4 pitanja)

Svaki unos sadrži:
  - question:          upit koji se šalje RAG sustavu
  - ground_truth:      točan odgovor poznat iz dokumenta (referentni odgovor)
                       prosljeđuje se kao references za Context recall za RAGAS
  - reference_context: ručno izdvojen „zlatni" odlomak/sažetak iz dokumenta koji
                       sadržava odgovor. NIJE ulaz u RAGAS — zadržan je kao
                       dokumentacija i podloga za ručnu provjeru dohvata
  - document:          naziv izvornog dokumenta
  - category:          kategorija dokumenta
"""

EVAL_DATASET = [

    # A) ZAKON O OIE I VISOKOUČINKOVITOJ KOGENERACIJI
   {
        "id": "ZOI-01",  # čl. 7. st. 1. 
        "question": "Koji je nacionalni cilj udjela obnovljivih izvora energije u ukupnoj potrošnji energije u Republici Hrvatskoj do 2030. godine?",
        "ground_truth": (
            "Nacionalni cilj korištenja energije iz obnovljivih izvora energije iznosi najmanje "
            "42,5 % obnovljivih izvora energije u konačnoj brutopotrošnji energije do 2030. "
            "godine u Republici Hrvatskoj."
        ),
        "reference_context": (
            "Korištenjem energije iz obnovljivih izvora energije ostvaruju se interesi Republike "
            "Hrvatske u području energetike, osobito u smislu ostvarenja nacionalnog cilja "
            "korištenja energije iz obnovljivih izvora energije od najmanje 42,5 % obnovljivih "
            "izvora energije u konačnoj brutopotrošnji energije do 2030. godine u Republici "
            "Hrvatskoj. (članak 7. stavak 1.)"
        ),
        "document": "Zakon o obnovljivim izvorima energije i visokoučinkovitoj kogeneraciji",
        "category": "Zakonodavstvo",
    },
    {
        "id": "ZOI-02",
        "question": "Što je tržišna premija i kako se izračunava za postrojenja na obnovljive izvore energije?",
        "ground_truth": (
            "Tržišna premija (TPi) je poticaj koji operator tržišta energije isplaćuje "
            "povlaštenom proizvođaču za neto isporučenu električnu energiju. Računa se kao "
            "razlika između referentne vrijednosti električne energije (RV), utvrđene ugovorom "
            "o tržišnoj premiji, i referentne tržišne cijene električne energije (TCi): "
            "TPi = RV − TCi."
    ),
        "reference_context": (
            "Članak 21. definira sustav poticanja tržišnom premijom, a članak 23. stavak 2. "
            "propisuje formulu TPi = RV − TCi, gdje je RV referentna vrijednost električne "
            "energije utvrđena ugovorom o tržišnoj premiji, a TCi referentna tržišna cijena "
            "električne energije u obračunskom razdoblju. Zajamčena otkupna cijena zaseban je "
            "mehanizam uređen člancima 24–26."
    ),
        "document": "Zakon o obnovljivim izvorima energije i visokoučinkovitoj kogeneraciji",
        "category": "Zakonodavstvo",
    },
    {
        "id": "ZOI-03",
        "question": "Koliko najdulje smije trajati upravni postupak izdavanja dozvola za izgradnju postrojenja OIE pred prvostupanjskim javnopravnim tijelom?",
        "ground_truth": (
            "Upravni postupak izdavanja dozvola za proizvodna postrojenja iz obnovljivih izvora "
            "ne smije trajati dulje od dvije godine pred prvostupanjskim javnopravnim tijelom "
            "(članak 29. stavak 11.). Za postrojenja kapaciteta električne energije ispod 150 kW "
            "rok je jedna godina (članak 29. stavak 12.). U opravdanim slučajevima izvanrednih "
            "okolnosti ili više sile navedeni se rokovi mogu produljiti za najviše jednu godinu."
    ),
        "reference_context": (
            "Članci 29–31 uređuju organizaciju i rokove postupaka izdavanja dozvola. Članak 29. "
            "stavak 11. propisuje najdulji rok od dvije godine za prvostupanjski postupak, stavak "
            "12. rok od jedne godine za postrojenja ispod 150 kW, a stavci 11., 12. i 14. mogućnost "
            "produljenja za najviše jednu godinu u izvanrednim okolnostima."
    ),
        "document": "Zakon o obnovljivim izvorima energije i visokoučinkovitoj kogeneraciji",
        "category": "Zakonodavstvo",
    },
    {
        "id": "ZOI-04",
        "question": "Po kojoj su reguliranoj otkupnoj cijeni opskrbljivači električne energije dužni preuzeti udio neto isporučene električne energije povlaštenih proizvođača?",
        "ground_truth": (
            "Opskrbljivači električne energije dužni su preuzeti udio, izražen u postotku, u neto "
            "isporučenoj električnoj energiji povlaštenih proizvođača, uključivo pravo na jamstvo "
            "podrijetla, po reguliranoj otkupnoj cijeni koja iznosi 0,055744 EUR/kWh "
            "(članak 46. stavak 1.)."
        ),
        "reference_context": (
            "Članak 46. stavak 1. propisuje obvezu opskrbljivača da preuzmu udio u neto isporučenoj "
            "električnoj energiji povlaštenih proizvođača po reguliranoj otkupnoj cijeni od "
            "0,055744 EUR/kWh. Članci 43–49 uređuju širi mehanizam prikupljanja i raspodjele "
            "sredstava za isplatu poticaja."
        ),
        "document": "Zakon o obnovljivim izvorima energije i visokoučinkovitoj kogeneraciji",
        "category": "Zakonodavstvo",
    },

    # B) PRIRUČNIK O POSTUPCIMA ISHOĐENJA DOZVOLA
   {
        "id": "PRI-01",  # str. 15–22 / pdf 19–26 (chunk 5.4) 
        "question": "Koji su ključni dionici u postupku ishođenja dozvola za izgradnju postrojenja za proizvodnju električne energije iz obnovljivih izvora?",
        "ground_truth": (
            "U postupcima ishođenja dozvola sudjeluju dionici na državnoj te lokalnoj i "
            "područnoj (regionalnoj) razini. Na državnoj razini to su MINGO, MZOZT, MPGI, "
            "MPS, HERA, HROTE, HOPS, HEP ODS, AZU, Hrvatske vode i Hrvatske šume. Jedinice "
            "lokalne i područne (regionalne) samouprave izdaju lokacijske i građevinske "
            "dozvole, a sudjeluju i područni uredi za katastar i zemljišnoknjižni odjeli. "
            "Uz njih sudjeluju ovlašteni članovi strukovnih komora (građevinarstva, "
            "arhitekture, elektrotehnike, geodezije, strojarstva). Prema Zakonu o gradnji "
            "ključni sudionici u gradnji su investitor, projektant, izvođač, nadzorni "
            "inženjer i revident."
        ),
        "reference_context": (
            "Na državnoj razini to su tijela državne uprave, druga državna tijela, agencije, "
            "operatori i druga trgovačka društva u državnom vlasništvu: Ministarstvo "
            "gospodarstva (MINGO), Ministarstvo zaštite okoliša i zelene tranzicije (MZOZT), "
            "Ministarstvo prostornoga uređenja, graditeljstva i državne imovine (MPGI), "
            "Ministarstvo poljoprivrede (MPS), HERA, HROTE, HOPS, HEP ODS, AZU, Hrvatske vode "
            "i Hrvatske šume. Jedinice lokalne i područne (regionalne) samouprave izdaju "
            "lokacijske i građevinske dozvole. Prema ZOG-u ključni sudionici u gradnji su: "
            "investitor, projektant, izvođač, nadzorni inženjer i revident."
        ),
        "document": "Priručnik o postupcima ishođenja dozvola za izgradnju proizvodnog postrojenja iz obnovljive energije",
        "category": "Procedure i dozvole",
    },
    {
        "id": "PRI-02",  # str. 83–85 / pdf 87–89 (chunk 5.5) 
        "question": "Koji je postupak stjecanja statusa povlaštenog proizvođača električne energije iz obnovljivih izvora?",
        "ground_truth": (
            "Nakon ishođenja uporabne dozvole potrebno je od HERA-e ishoditi dozvolu za "
            "obavljanje energetske djelatnosti proizvodnje, a potom od HERA-e i rješenje o "
            "stjecanju statusa povlaštenog proizvođača električne energije, koje se izdaje na "
            "rok važenja od 25 godina. Dozvolu za obavljanje energetske djelatnosti nije "
            "potrebno ishoditi ako je zbroj instaliranih snaga svih postrojenja investitora "
            "do uključivo 500 kW i ako se postrojenje nalazi iza obračunskog mjernog mjesta "
            "aktivnog kupca. Proizvodna postrojenja koja su jednostavne građevine stječu "
            "status povlaštenog proizvođača temeljem ZOIEVUK-a i ne trebaju ishoditi rješenje "
            "HERA-e, osim ako žele sudjelovati u sustavu jamstva podrijetla energije."
        ),
        "reference_context": (
            "Po ishođenju dozvole za obavljanje energetske djelatnosti potrebno je od HERA-e "
            "ishoditi rješenje o statusu povlaštenog proizvođača. Rješenje o stjecanju statusa "
            "povlaštenog proizvođača električne energije izdaje HERA na zahtjev stranke čije "
            "proizvodno postrojenje zadovoljava uvjete, na rok važenja od 25 godina. "
            "Proizvodna postrojenja koja su jednostavne građevine stječu status povlaštenosti "
            "temeljem ZOIEVUK-a te ne trebaju ishoditi rješenje od HERA-e, osim u slučaju ako "
            "se s takvim postrojenjem želi sudjelovati u sustavu jamstva podrijetla energije."
        ),
        "document": "Priručnik o postupcima ishođenja dozvola za izgradnju proizvodnog postrojenja iz obnovljive energije",
        "category": "Procedure i dozvole",
    },
    {
        "id": "PRI-03",  # str. 86–97 / pdf 90–101 (chunk 5.6) 
        "question": "Kako funkcionira model samoopskrbe (vlastita potrošnja) za sunčane elektrane i kako se obračunavaju viškovi proizvedene energije?",
        "ground_truth": (
            "U modelu samoopskrbe elektrana je priključena iza obračunskog mjernog mjesta "
            "kupca, pa se proizvedena energija najprije troši za vlastite potrebe, a tek se "
            "višak predaje u mrežu. Opskrbljivač je dužan preuzimati viškove kupaca s "
            "vlastitom proizvodnjom koji zadovoljavaju uvjete iz članka 51. ZOIEVUK-a. "
            "Zajamčena minimalna otkupna cijena viškova Ci obračunava se kao Ci = 0,9 × PKCi "
            "(kada je preuzeta energija veća ili jednaka isporučenoj), pri čemu je PKCi "
            "prosječna jedinična cijena koju kupac plaća opskrbljivaču. Obračunsko razdoblje "
            "je kalendarski mjesec, a mjerenje se provodi dvosmjernim intervalnim brojilom."
        ),
        "reference_context": (
            "Kod modela vlastite proizvodnje elektrana je priključena iza obračunskog mjernog "
            "mjesta kupca te se proizvedena električna energija najprije troši za pokrivanje "
            "vlastitih potreba, a tek višak se predaje u mrežu. ZOIEVUK-om je kupcu s "
            "vlastitom proizvodnjom zagarantirana minimalna otkupna cijena viškova Ci: "
            "Ci = 0,9 × PKCi ako vrijedi Epi ≥ Eii. Obračunsko razdoblje za kupca s vlastitom "
            "proizvodnjom je kalendarski mjesec."
        ),
        "document": "Priručnik o postupcima ishođenja dozvola za izgradnju proizvodnog postrojenja iz obnovljive energije",
        "category": "Procedure i dozvole",
    },
    {
        "id": "PRI-04",  # str. 117–121 / pdf 121–125 (chunk 5.7)
        "question": "Koja je razlika između integriranih i neintegriranih sunčanih elektrana pri ishođenju dozvola?",
        "ground_truth": (
            "Neintegrirane sunčane elektrane su elektrane izgrađene kao samostojeće "
            "građevine, dok su integrirane one smještene na dijelovima građevina (krovovi, "
            "pokrovi, sjenila, balkoni, terase, nadstrešnice, fasade, prozori i dr.) ili na "
            "infrastrukturnim objektima. Razlika utječe na postupak: za neintegrirane se "
            "redom provode izrada idejnog rješenja, ishođenje energetskog odobrenja i "
            "elektroenergetske suglasnosti te građenje s uporabnom dozvolom. Za integrirane "
            "postupak započinje zahtjevom za priključenje operatoru distribucijskog sustava, "
            "nije potrebna građevinska ni uporabna dozvola (uz stručni nadzor i prijavu "
            "početka građenja), a građe se u skladu s glavnim projektom."
        ),
        "reference_context": (
            "Neintegrirane sunčane elektrane su sunčane elektrane izgrađene kao samostojeće "
            "građevine. Integriranom sunčanom elektranom smatra se sunčana elektrana "
            "smještena na dijelovima građevina kao što su krovovi, pokrovi, sjenila, balkoni, "
            "terase, nadstrešnice, balustrade, fasade, prozori, vrata i druge površine "
            "građevina ili na transformatorskim stanicama, nadstrešnicama, mostovima i drugim "
            "infrastrukturnim objektima. Kod izgradnje integriranih sunčanih elektrana nije "
            "potrebno ishođenje građevinske dozvole; za integrirane elektrane ne izdaje se "
            "uporabna dozvola, ali je potreban stručni nadzor građenja i prijava početka "
            "građenja sukladno ZOG-u."
        ),
        "document": "Priručnik o postupcima ishođenja dozvola za izgradnju proizvodnog postrojenja iz obnovljive energije",
        "category": "Procedure i dozvole",
    },
    # C) METODOLOGIJA ENERGETSKOG PREGLEDA ZGRADA 2021
    {
        "id": "MET-01",
        "question": "Što je energetski pregled zgrade i koji su mu ciljevi prema Metodologiji 2021?",
        "ground_truth": (
            "Energetski pregled zgrade sustavni je postupak prikupljanja i analize "
            "podataka o energetskim karakteristikama zgrade s ciljem utvrđivanja "
            "energetskog razreda i preporuka za poboljšanje energetske učinkovitosti. "
            "Ciljevi i svrha opisani su na stranicama 1–15 Metodologije."
        ),
        "reference_context": (
            "Stranice 1–15 Metodologije 2021 sadrže uvodni dio s definicijom energetskog "
            "pregleda, svrhom, ciljevima i opisom tijeka postupka energetskog "
            "certificiranja zgrada."
        ),
        "document": "Metodologija provođenja energetskog pregleda zgrada 2021",
        "category": "Energetska učinkovitost",
    },
    {
        "id": "MET-02",  # chunk 1.0, str. 15 / pdf 28 
        "question": "Što mora sadržavati plan mjerenja koji se izrađuje u sklopu pripreme kontrolnih mjerenja pri energetskom pregledu zgrade?",
        "ground_truth": (
            "Plan mjerenja ključni je dokument pripreme kontrolnih mjerenja koji mora "
            "sadržavati odgovore na pitanja: tko mjeri, gdje se mjeri, koliko traje mjerenje, "
            "tko je od strane korisnika odobrio mjerenje, s kojom se mjernom opremom mjeri "
            "te tko kontrolira mjerenje. Plan mjerenja mora biti sastavni dio dokumentacije "
            "koja se zajedno s rezultatima mjerenja predaje korisniku."
        ),
        "reference_context": (
            "Priprema mjerenja mora uključivati izradu plana mjerenja. Plan mjerenja mora "
            "sadržavati odgovore na pitanja: Tko mjeri? Gdje se mjeri? Koliko traje mjerenje? "
            "Tko je od strane korisnika odobrio mjerenje? S kojom mjernom opremom se vrši "
            "mjerenje? Tko kontrolira mjerenje? Plan mjerenja mora biti sastavni dio "
            "dokumentacije koja se zajedno s rezultatima mjerenja predaje korisniku."
        ),
        "document": "Metodologija provođenja energetskog pregleda zgrada 2021",
        "category": "Energetska učinkovitost",
    },
   {
        "id": "MET-03",  # chunk 1.3, str. 197 / pdf 210 
        "question": "Koje mjere povećanja energetske učinkovitosti obuhvaća građevinska (vanjska) ovojnica zgrade prema Metodologiji 2021?",
        "ground_truth": (
            "Mjere energetske učinkovitosti na vanjskoj ovojnici zgrade obuhvaćaju toplinsku "
            "izolaciju vanjskih zidova, ravnih krovova, kosih krovova i stropova prema "
            "provjetravanom tavanu, zidova prema negrijanim prostorima, stropova iznad "
            "vanjskog zraka, zidova i podova prema tlu te toplinskih mostova; brtvljenje i "
            "izmjenu stolarije/bravarije; toplinsku izolaciju kutija za rolete te ugradnju "
            "zaštite od Sunčevog zračenja. To su sve mjere na građevinskoj ovojnici koje "
            "doprinose smanjenju potrebne energije za grijanje i hlađenje zgrade."
        ),
        "reference_context": (
            "Mjere energetske učinkovitosti kod vanjske ovojnice zgrade: toplinska izolacija "
            "vanjskih zidova, ravnih krovova, stropova prema provjetravanom tavanu i kosih "
            "krovova, zidova prema negrijanim prostorima, stropova iznad vanjskog zraka; "
            "brtvljenje stolarije/bravarije; toplinska izolacija zidova i podova prema tlu, "
            "stropova i podova negrijanih prostorija prema grijanima, toplinskih mostova; "
            "izmjena stolarije/bravarije; toplinska izolacija kutija za rolete; ugradnja "
            "zaštite od Sunčevog zračenja. Mjere povećanja energetske učinkovitosti vanjske "
            "ovojnice sve su mjere na građevinskoj ovojnici zgrade koje doprinose smanjenju "
            "potrebne energije za grijanje i hlađenje zgrade."
        ),
        "document": "Metodologija provođenja energetskog pregleda zgrada 2021",
        "category": "Energetska učinkovitost",
    },
    {
        "id": "MET-04",  # chunk 1.5, str. 272 / pdf 303 
        "question": "Za koje sustave i uz koji prag nazivne snage je prema Metodologiji 2021 obvezan redoviti pregled sustava grijanja i hlađenja prostora?",
        "ground_truth": (
            "Redoviti pregled sustava grijanja prostora obvezno se provodi za sve dostupne "
            "dijelove centralnog sustava grijanja ili kombiniranog sustava grijanja i prisilne "
            "ventilacije/klimatizacije nazivne toplinske snage veće od 70 kW (izvor toplinske "
            "energije, cirkulacijske crpke i sustav regulacije). Redoviti pregled sustava "
            "hlađenja prostora obvezan je za centralne sustave hlađenja nazivne rashladne "
            "snage veće od 70 kW. Kod više centralnih izvora koji rade zajednički, ukupna se "
            "nazivna snaga računa kao zbroj pojedinačnih snaga."
        ),
        "reference_context": (
            "Redoviti pregled sustava grijanja prostora provodi se obvezno za sve dostupne "
            "dijelove centralnog sustava grijanja prostora ili kombiniranog centralnog sustava "
            "grijanja i prisilne ventilacije/klimatizacije prostora, nazivne toplinske snage "
            "veće od 70 kW, kao što su izvor toplinske energije (generator topline), "
            "cirkulacijske crpke i sustav regulacije. Redoviti pregled sustava hlađenja "
            "prostora obvezno se provodi za sve dostupne dijelove centralnog sustava hlađenja "
            "prostora ili kombiniranog sustava hlađenja i prisilne ventilacije/klimatizacije "
            "prostora, nazivne rashladne snage veće od 70 kW. U slučaju više centralnih izvora "
            "koji zajednički rade, ukupna nazivna toplinska snaga računa se kao zbroj "
            "pojedinačnih nazivnih toplinskih snaga."
        ),
        "document": "Metodologija provođenja energetskog pregleda zgrada 2021",
        "category": "Energetska učinkovitost",
    },

   # D) TEHNIČKI PROPIS O SUSTAVIMA VENTILACIJE
    {
        "id": "VEN-01",  # chunk 2.1, čl. 9 (str. 8) — tehnička svojstva
        "question": "Koje uvjete moraju ispunjavati tehnička svojstva sustava ventilacije prema Tehničkom propisu o sustavima ventilacije?",
        "ground_truth": (
            "Tehnička svojstva sustava ventilacije moraju biti takva da tijekom trajanja "
            "zgrade podnesu utjecaje uobičajene uporabe i okoliša tako da se u slučaju "
            "požara spriječi širenje vatre i dima unutar zgrade i na susjedne građevine, "
            "da se zadovolje uvjeti kvalitete zraka i spriječi sakupljanje vlage, da se "
            "izbjegnu moguće ozljede korisnika te da razina buke ne ugrožava zdravlje i "
            "osigurava noćni mir. Kod sustava s procesom grijanja zraka, djelomične "
            "klimatizacije i klimatizacije dodatno se moraju osigurati zadovoljavajući "
            "toplinski uvjeti uz potrošnju energije jednaku ili nižu od propisane."
        ),
        "reference_context": (
            "Tehnička svojstva sustava ventilacije moraju biti takva da podnesu sve "
            "utjecaje uobičajene uporabe i okoliša tako da: se u slučaju požara spriječi "
            "širenje vatre i dima unutar zgrade odnosno na susjedne građevine; se zadovolje "
            "uvjeti kvalitete zraka te spriječi sakupljanje vlage; se izbjegnu moguće "
            "ozljede korisnika zgrade; razina buke ne ugrožava zdravlje i osigurava noćni "
            "mir. Sustavi s procesom grijanja zraka, djelomične klimatizacije i klimatizacije "
            "moraju dodatno osigurati zadovoljavajuće toplinske uvjete uz potrošnju energije "
            "jednaku ili nižu od potrošnje određene posebnim propisom."
        ),
        "document": "Tehnički propis o sustavima ventilacije, djelomične klimatizacije i klimatizacije zgrada",
        "category": "Tehnički propisi",
    },
    {
        "id": "VEN-02",  # chunk 2.1, čl. 17 (str. 11–13) — sadržaj strojarskog projekta
        "question": "Što mora sadržavati strojarski projekt sustava ventilacije kao sastavni dio glavnog projekta zgrade prema Tehničkom propisu?",
        "ground_truth": (
            "Strojarski projekt sustava mora u tehničkom opisu sadržavati opis rada sustava "
            "i procesa gospodarenja energijom, utjecaj sustava na okoliš (buka, vibracije, "
            "zagađenost, povrat topline), rješenje ugradnje i pričvršćenja na nosivu "
            "konstrukciju, određivanje dijelova za sprječavanje širenja vatre i dima te "
            "uvjete održavanja. U proračunima mora sadržavati termodinamički proračun "
            "toplinskih opterećenja, proračun tehničkih svojstava komponenata, hidrauličke "
            "proračune vodnih i zračnih sustava te proračun sustava povrata energije za "
            "uređaje s količinom vanjskog zraka većom od 2 500 m³/h."
        ),
        "reference_context": (
            "Strojarski projekt sustava koji je sastavni dio glavnog projekta zgrade mora "
            "sadržavati osobito u tehničkom opisu: opis rada sustava i način uporabe te opis "
            "procesa gospodarenja energijom; utjecaj sustava na okoliš (buka, vibracije, "
            "zagađenost, povrat topline); opis rješenja ugradnje, pričvršćenja i ovješenja "
            "na nosivu konstrukciju; određivanje svojstava i položaja dijelova namijenjenih "
            "sprječavanju širenja vatre i dima; uvjete održavanja. U proračunima (ovisno o "
            "sustavu): termodinamički proračun toplinskih opterećenja zgrade (ljeto/zima), "
            "proračun tehničkih svojstava komponenata, hidraulički proračun vodnih i zračnih "
            "sustava, proračun sustava povrata energije za sve uređaje s količinom vanjskog "
            "zraka većom od 2 500 m³/h."
        ),
        "document": "Tehnički propis o sustavima ventilacije, djelomične klimatizacije i klimatizacije zgrada",
        "category": "Tehnički propisi",
    },
   {
        "id": "VEN-03",  # chunk 2.2, Prilog B B.2.1.4 (str. 22–23) — nadzorni inženjer prije izvođenja
        "question": "Što nadzorni inženjer mora provjeriti neposredno prije početka izvođenja sustava ventilacije prema Prilogu B Tehničkog propisa?",
        "ground_truth": (
            "Nadzorni inženjer mora neposredno prije početka izvođenja sustava provjeriti "
            "postoji li isprava o sukladnosti u skladu s posebnim propisima za građevne, "
            "strojarske i druge proizvode koji se ugrađuju te jesu li iskazana svojstva "
            "sukladna zahtjevima iz projekta zgrade; provjeriti jesu li ti proizvodi "
            "ugrađeni u skladu s projektom zgrade i/ili tehničkom uputom za ugradnju i "
            "uporabu, s Prilogom A propisa i odredbama posebnih propisa; te dokumentirati "
            "nalaze svih provedenih provjera zapisom u građevinski dnevnik."
        ),
        "reference_context": (
            "Nadzorni inženjer neposredno prije početka izvođenja sustava mora: "
            "a) provjeriti postoji li isprava o sukladnosti u skladu s posebnim propisima "
            "za građevne, strojarske i druge proizvode koji se ugrađuju u sustave i jesu li "
            "iskazana svojstva sukladna zahtjevima iz projekta zgrade, "
            "b) provjeriti jesu li građevni, strojarski i drugi proizvodi ugrađeni u skladu "
            "s projektom zgrade i/ili tehničkom uputom za ugradnju i uporabu sustava, s "
            "Prilogom »A« ovoga propisa i odredbama posebnih propisa, "
            "c) dokumentirati nalaze svih provedenih provjera zapisom u građevinski dnevnik."
        ),
        "document": "Tehnički propis o sustavima ventilacije, djelomične klimatizacije i klimatizacije zgrada",
        "category": "Tehnički propisi",
    },
    {
        "id": "VEN-04",  # chunk 2.2, Prilog B (str. 24, B.3.2–B.3.3) — održavanje/pregledi
        "question": "Koliko se često provode redoviti pregledi sustava ventilacije tijekom održavanja i za koje je sustave ispitivanje obvezno prema Tehničkom propisu?",
        "ground_truth": (
            "Učestalost redovitih pregleda u svrhu održavanja sustava provodi se sukladno "
            "zahtjevima projekta zgrade, ali ne rjeđe od jednom godišnje. Ispitivanje "
            "sustava tijekom održavanja obvezno je za sustave ogrjevnog učina preko 20 kW "
            "i rashladnog učina preko 12 kW. Prigodom pregleda sustav se obvezno čisti i "
            "dezinficira."
        ),
        "reference_context": (
            "Ispitivanje sustava tijekom održavanja obavezno je za sustave ogrijevnog učina "
            "preko 20 kW i rashladnog učina preko 12 kW. Učestalost redovitih pregleda u "
            "svrhu održavanja sustava provodi se sukladno zahtjevima projekta zgrade, ali "
            "ne rjeđe od jednom godišnje. Prigodom pregleda sustava sustav se obvezno čisti "
            "i dezinficira."
        ),
        "document": "Tehnički propis o sustavima ventilacije, djelomične klimatizacije i klimatizacije zgrada",
        "category": "Tehnički propisi",
    },

   # E) JAVNI POZIV EnU-6/25 — FOTONAPONSKE ELEKTRANE
    {
        "id": "JPO-01",  # odjeljak III – Korisnici (str. 2–3)
        "question": "Tko može biti korisnik sredstava i koje uvjete mora ispuniti prijavitelj na Javni poziv EnU-6/25?",
        "ground_truth": (
            "Korisnici sredstava mogu biti fizičke osobe – građani, vlasnici ili "
            "suvlasnici obiteljske kuće koji u trenutku puštanja fotonaponske elektrane "
            "u pogon imaju prebivalište na adresi i mjestu kuće. Prijavitelj mora uložiti "
            "vlastita sredstva, nemati dospjelih nepodmirenih dugovanja prema Fondu te "
            "dostaviti svu obveznu dokumentaciju sukladno Pozivu."
        ),
        "reference_context": (
            "Korisnici sredstava u smislu ovog Poziva su fizičke osobe – građani, vlasnici "
            "ili suvlasnici obiteljske kuće koji u trenutku puštanja fotonaponske elektrane "
            "u pogon imaju prebivalište na adresi i mjestu kuće."
        ),
        "document": "Javni poziv za poticanje ugradnje fotonaponskih elektrana u obiteljskim kućama (EnU-6/25)",
        "category": "Financiranje i potpore",
    },
    {
        "id": "JPO-02",  # odjeljak I – Predmet (str. 1)
        "question": "Koji je minimalni stupanj korisnog djelovanja fotonaponskih modula za prihvatljivost sufinanciranja prema Javnom pozivu EnU-6/25?",
        "ground_truth": (
            "Za sufinanciranje su prihvatljivi ugrađeni fotonaponski sunčani moduli "
            "stupnja korisnog djelovanja od najmanje 18%."
        ),
        "reference_context": (
            "Za sufinanciranje su prihvatljive isključivo fotonaponske elektrane koje su "
            "ugrađene u periodu od 1. siječnja 2024. godine do 31. prosinca 2024. godine i "
            "puštene u pogon u periodu od 1. siječnja 2024. godine do 6. lipnja 2025. godine, "
            "stupnja korisnog djelovanja ugrađenih fotonaponskih sunčanih modula najmanje 18%."
        ),
        "document": "Javni poziv za poticanje ugradnje fotonaponskih elektrana u obiteljskim kućama (EnU-6/25)",
        "category": "Financiranje i potpore",
    },
    {
        "id": "JPO-03",  # odjeljak IV – Sredstva Fonda (str. 3)
        "question": "Koji je iznos potpore po kW i maksimalni udio sufinanciranja opravdanih troškova prema Javnom pozivu EnU-6/25?",
        "ground_truth": (
            "Sredstva se dodjeljuju u visini od 600 eura po kW nazivne snage ugrađene "
            "fotonaponske elektrane, a najviše do 50% opravdanih troškova Projekta."
        ),
        "reference_context": (
            "Ukupno raspoloživ iznos sredstava Fonda po ovom Pozivu je 7.500.000,00 eura. "
            "Sredstva se dodjeljuju u visini od 600 eura po kW nazivne snage ugrađene "
            "fotonaponske elektrane, a najviše do 50% opravdanih troškova Projekta."
        ),
        "document": "Javni poziv za poticanje ugradnje fotonaponskih elektrana u obiteljskim kućama (EnU-6/25)",
        "category": "Financiranje i potpore",
    },
    {
        "id": "JPO-04",  # odjeljak VIII – Isplata (str. 8)
        "question": "U kojem roku i na koji način Fond isplaćuje odobrena sredstva korisniku prema Javnom pozivu EnU-6/25?",
        "ground_truth": (
            "Fond odobrena sredstva isplaćuje jednokratno na IBAN korisnika, u roku od "
            "30 dana od dana stupanja na snagu Odluke o odabiru korisnika i dodjeli sredstava."
        ),
        "reference_context": (
            "Odobrena sredstva Fond će isplatiti jednokratno na IBAN korisnika, u roku od "
            "30 dana od dana stupanja na snagu Odluke."
        ),
        "document": "Javni poziv za poticanje ugradnje fotonaponskih elektrana u obiteljskim kućama (EnU-6/25)",
        "category": "Financiranje i potpore",
    },
]

if __name__ == "__main__":
    print(f"Ukupan broj pitanja: {len(EVAL_DATASET)}\n")
    kategorije = {}
    for item in EVAL_DATASET:
        kat = item["category"]
        kategorije[kat] = kategorije.get(kat, 0) + 1

    print("Raspodjela po kategorijama:")
    for kat, broj in kategorije.items():
        print(f"  {kat}: {broj} pitanja")

    print("\nPrimjer prvog pitanja:")
    p = EVAL_DATASET[0]
    print(f"  ID: {p['id']}")
    print(f"  Pitanje: {p['question']}")
    print(f"  Točan odgovor: {p['ground_truth'][:80]}...")
