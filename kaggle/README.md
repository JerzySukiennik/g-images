# Trening G-Images na Kaggle — instrukcja

Ten sam plan co przy [[microg|MicroG]]: **30 h GPU tygodniowo za darmo** (T4 ×2), sesja ucinana po 12 h, wszystko poniżej zbudowane pod wznawianie.

Czego potrzebujesz: konta na kaggle.com z **zweryfikowanym numerem telefonu**.

---

## Dlaczego nie wysyłamy danych z domu

`timbrooks/instructpix2pix-clip-filtered` jest ogromny — bierzemy tylko pierwsze **60k** par w 128px (podniesione z 20k 2026-07-22, patrz "Ile danych i dlaczego" niżej), ale nawet to lepiej ściągnąć na szybkim łączu Kaggle niż z domu. **Pobieramy i pakujemy dane na Kaggle**, raz, zapisujemy jako Dataset. Każda kolejna sesja treningowa startuje z gotowych binariów w kilka sekund.

---

## Krok 1 — przygotowanie danych (raz, ~1.5-2h dla 60k par)

1. **New Notebook**
2. Prawy panel → **Accelerator: None**, **Internet: On**
3. *(opcjonalnie, przyspiesza pobieranie)* **Add-ons → Secrets → Add secret**, nazwa `HF_TOKEN`, wartość = Twój token z HuggingFace
4. Wklej zawartość [`01-prep.py`](01-prep.py) do komórki i uruchom
5. **Save Version → Save & Run All**, poczekaj aż się wykona
6. Na stronie wersji: **Output → New Dataset**, nazwij **`gedit-data`**

Powstaną `gedit_images.bin` (obrazy przed/po, uint8), `gedit_text.bin` (embeddingi CLIP instrukcji edycji, float32) i `gedit_meta.json`.

## Krok 1.5 — po zmianie architektury (FiLM → cross-attention, 2026-07-22): przelicz teksty

Jeśli masz już `gedit-data` (obrazy pobrane), **nie trzeba pobierać ich drugi raz** — sam koder tekstu się zmienił (pełna sekwencja tokenów zamiast jednego wektora, patrz `model/clip_encoder.py`). Wklej [`03-reencode-text.py`](03-reencode-text.py) w nowym notebooku (Accelerator: None, Internet: On, **Add Input** → istniejący `gedit-data`), uruchom (kilka minut, nie godziny), zapisz Output jako Dataset `gedit-data` (nadpisując stary).

Stary checkpoint (FiLM) **nie jest kompatybilny** z nową architekturą — nie da się go wznowić, kolejny krok to trening od zera.

## Krok 2 — trening (czas nieznany do pierwszego pomiaru na nowej architekturze)

1. **New Notebook**
2. **Accelerator: GPU T4 x2**, **Internet: On**, **Persistence: Variables and Files**
3. **Add Input** → dataset `gedit-data` (już z przeliczonymi tekstami)
4. Wklej [`02-train.py`](02-train.py) i uruchom
5. **Save Version → Save & Run All (Commit)** — nie zwykłe odpalenie komórki. Batch-commit wypełnia zakładkę **Output** automatycznie po zakończeniu; interaktywna sesja ("Edit") tego nie robi, trzeba wtedy ręcznie szukać pliku przez panel plików po prawej (Notebook → Output → rozwiń `/kaggle/working` → `run/` → `ckpt.pt`) — działa, ale niepotrzebnie karkołomne.
6. Po zakończeniu (albo ucięciu na 12h): **Output → New Dataset**, nazwij **`gedit-ckpt`**

Checkpoint zapisuje się co 200 kroków niezależnie od tego, ile `STEPS` ustawiono w skrypcie — **STEPS to sufit, nie cel**. Można pobrać `ckpt.pt` i zatrzymać sesję w dowolnym momencie, jak tylko `runtime/sample_check.py` (patrz niżej) pokaże wystarczająco dobry wynik.

## Krok 3 — wznowienie (jeśli 12 h nie starczyło, albo chcesz trenować dalej)

Nowy notebook jak w kroku 2, ale **Add Input** dwa razy: `gedit-data` **oraz** `gedit-ckpt`. Skrypt sam wykryje `ckpt.pt` i podejmie od ostatniego kroku — razem z momentami Adama, więc bez skoku loss.

Po każdej sesji nadpisuj `gedit-ckpt` nowym outputem (**Output → Update Dataset**). Jeśli checkpoint już osiągnął `STEPS` z aktualnej wersji skryptu, wznowienie nic nie zrobi (pętla `while step < max_steps` od razu się kończy) — trzeba najpierw podnieść `STEPS` w `02-train.py`.

---

## Ile danych i dlaczego (2026-07-22)

Pierwszy realny trening: **N=20000 par, 40000 kroków zaplanowane** — ale to by wyszło na **~65 epok** (powtórzeń) tych samych 19700 par treningowych, bez żadnej augmentacji danych. Duże ryzyko zapamiętywania konkretnych przykładów zamiast uczenia się ogólnych wzorców edycji. Podniesione do **N=60000** (3x), **STEPS=28000** → ~15 epok na trzykrotnie bogatszym zestawie, zdrowszy stosunek powtórzeń do różnorodności. Strumień z HuggingFace jest deterministyczny, więc pierwsze 20000 par nowego zestawu to dokładnie to samo co poprzednio, plus 40000 nowych — nie tracimy nic z wcześniejszego pobierania w sensie treści, choć `01-prep.py` i tak pobiera wszystko od nowa (nie ma resume na poziomie pojedynczych par).

## Sprawdzanie jakości wizualnie, nie tylko po loss

**Loss z logu bywa mylący.** Przy pierwszym treningu (20k par) pojedynczy-batch loss spłaszczył się już koło kroku 300-760, ale jakość obrazków z `runtime/sample_check.py` dalej realnie rosła aż do kroku 8000 — spłaszczenie lossu było artefaktem szumu pojedynczej próbki i "łatwych" dużych `t` w harmonogramie dyfuzji (patrz komentarz w `model/unet.py`/`scheduler.py`), nie prawdziwym sufitem jakości. **Traktuj loss jako pomocniczy sygnał, nie ostateczny sędzia** — właściwa ocena to odpalenie `runtime/sample_check.py` na realnym checkpoincie i patrzenie na obrazki (SPEC.md #1: świadomie brak automatycznej metryki jakości).

Sposób odpalenia (lokalnie, po ściągnięciu `ckpt.pt` i datasetu z Kaggle):
```
python runtime/sample_check.py --data <prefix> --ckpt ckpt.pt --n 8 --out sample_check.png
```

## Jak sprawdzić, czy idzie dobrze

**Loss startowy to najważniejsza liczba w pierwszej minucie** — dla straty MSE na przewidywaniu szumu (wariancja jednostkowa) startowy loss powinien być **w okolicach 1.0**. Wartość dużo wyższa albo `nan` od razu = coś jest zepsute (zły zakres pikseli, zła normalizacja) — sprawdź przed inwestowaniem godzin GPU.

Val loss (log co `--eval-every`) powinien systematycznie spadać poniżej 1.0 w pierwszych ~500 krokach. Jeśli stoi w miejscu przez kilkaset kroków — learning rate za niski albo dane zepsute. Jeśli skacze do `nan` — zbij `--lr`. Poza tym wczesnym sanity-checkiem, dalszy postęp oceniaj wizualnie (patrz sekcja wyżej), nie po samej krzywej.

## Jak coś nie działa

- **`CUDA out of memory`** → zbij `BATCH` w `02-train.py`, podnieś `ACCUM` (ta sama liczba przykładów na krok)
- **DataParallel się sypie** → dopisz `"--single-gpu"` do listy `cmd` w `02-train.py`
- **Notebook nie widzi danych** → sprawdź, czy dataset jest faktycznie dodany w **Add Input**, nie tylko utworzony
- **Val loss dużo wyższy niż train loss** → 300 przykładów walidacyjnych (domyślne `--val-n`) to mało przy małym batchu, trochę szumu jest normalne
