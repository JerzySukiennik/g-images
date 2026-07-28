# G-Images — SPEC v1

Drugi model w rodzinie [[microg|MicroG]]: lokalny, w 100% własny model do edycji zdjęć, wywoływany głosem z [[gzowo-ai]] ("zrób mi zdjęcie i spraw żeby wyglądało jak X"), edytujący klatkę z kamery LOKALNIE (nie przez zewnętrzne API) i pokazujący wynik w oknie asystenta.

Status: **spec do zatwierdzenia, przed pierwszym kodem** (ta sama reguła co przy MicroG i gzowo-ai).

## 0. Nazwa i marka (2026-07-21)

**G-Images** — "G" (Gzowo, ten sam rdzeń co MicroG) + "edit" (dokładnie to, co model robi). Świadoma kolizja nazwy z linuksowym edytorem tekstu `gedit` — inny kontekst, mało prawdopodobne pomylenie w praktyce, zaakceptowana przez Jurka.

**Logo: monogram, ten sam szkielet geometrii co µG.** Lewy znak nie jest literą — to **crop-mark** (dwa przeciwległe narożniki L, uniwersalna ikona kadrowania/edycji zdjęć), stojący w miejscu µ dokładnie tak, jak µ stoi za słowem "micro". Zajmuje IDENTYCZNY bounding box co µ w MicroG (x 60-132, y 74-172), więc para czyta się na tej samej wadze optycznej i nie wymaga przeliczania transformu w ikonie. Prawa strona (G, promień 50, środek 200,90) jest **dosłownie tą samą ścieżką SVG** co w MicroG — ta sama grubość kreski 15, ta sama linia bazowa 140.

Źródła: `Design/gedit-{mark,mark-inverse,icon}.svg` w tym folderze, zbudowane wg dokładnie tego samego wzorca co `MicroG/Design/microg-*.svg` (skopiowana konstrukcja, podmieniony tylko lewy glif). Rasteryzacja do `.icns` — ten sam pipeline co MicroG (`Design/build-icons.sh` + `rasterize.html`/`raster-server.py`), do skopiowania i podmiany nazw przy realnym buildzie.

Slogan — do ustalenia razem z liczbą parametrów finalnego modelu (wzorem MicroG "100M parameters. 100% Gzowo."); robocza wersja: **"Small pixels. 100% Gzowo."**

## 1. Zakres funkcjonalny

- Edycje: **style artystyczne/filtry** (vintage, cyberpunk-glow, czarno-białe itd.) + **proste dodawanie/usuwanie obiektów** (inpainting).
- Komenda: dowolny opis słowny, mapowany na tekstowy embedding — bez zamkniętej listy nazwanych stylów.
- Każda edycja zawsze wychodzi od **świeżego zdjęcia z kamery** (bez łańcuchowania edycji na edycjach).
- Wynik zawsze się pokazuje, bez automatycznego odrzucania słabych wyników (ocena "na oko" przez Jurka) — na start.
- iMessage (wysyłanie zdjęcia do kogoś) — **odłożone**, dorabiamy po tym, jak core (kamera→edycja→podgląd) działa stabilnie.

## 2. Model — architektura i trening

- **W 100% własny model dyfuzyjny** (mała liczba kroków odszumiania), trenowany **od zera**, jak przy MicroG — ale świadomie w ograniczonej skali:
  - niska rozdzielczość treningowa/inference (64–128 px), upscaling do pełnej rozdzielczości **decyzja odłożona** do zobaczenia jakości pierwszych wyników (Lanczos vs gotowy model super-res).
  - koder tekstu (zamiana opisu słownego na embedding) — **gotowy, zamrożony pretrenowany CLIP text encoder**, NIE trenowany przez nas. Cały obrazowy U-Net/diffusion model, który faktycznie edytuje zdjęcie, jest w 100% własny i trenowany od zera — to standardowa praktyka (tak samo robi Stable Diffusion).
- **Dataset**: [huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered) (pary przed/po + instrukcja edycji, wygenerowane przez Prompt-to-Prompt). Dataset jest duży (setki tysięcy par) — bierzemy **mały podzbiór (rząd 10-30 tys. par)** w niskiej rozdzielczości, żeby trening był wykonalny w kilka-kilkanaście godzin, nie dni/tygodnie.
- **Trening: Kaggle** (darmowe GPU T4x2, limit ~30h/tydzień), ten sam workflow co przy MicroG — checkpointy + wznawianie sesji co ~12h.
- **Jakość**: brak automatycznej metryki na start, ocena wizualna przez Jurka po każdym checkpoincie (tak jak przy MicroG).

## 3. Inference i integracja z gzowo-ai

Zweryfikowane w realnym kodzie gzowo-ai (`~/Downloads/Claude/Projects/Gzowo AI/v1`):

- **Eksport**: PyTorch (trening) → **ONNX** → inference w bridge'u przez `onnxruntime-node`. Bridge to Node.js (porty 8787/8788), zero Pythona w runtime — `bridge/package.json` obecnie ma tylko `@google/genai`, trzeba dodać `onnxruntime-node`.
- **Rozszerzenie istniejącego trybu Vision** w gzowo-ai, nie nowy osobny tryb:
  - kamera: `js/vision/camera.js` (`cameraVision` singleton) — nowa funkcja `captureFrame()` (już istnieje) do jednorazowego zdjęcia, bez zmian w istniejącym 1fps live-feed do Gemini.
  - nowe narzędzie w `js/vision/vision-tools.js`, zarejestrowane przez `toolRouter.registerTool({name: 'edit_camera_photo', description, parameters}, handler)` — wzorzec identyczny jak istniejące `create_3d_model` (capture → POST do bridge → poll job → widget wyniku).
  - **wzorzec job-queue** z `bridge/model3d-jobs.js` (`createModelJob()` zwraca job id od razu, `setImmediate` robi robotę w tle, klient polluje `GET /.../jobs/:id`) — bo inference dyfuzyjny może zająć kilka-kilkanaście sekund na CPU/MPS Twojego Maca, nie może blokować requesta.
  - nowy plik `bridge/photo-edit.js` z handlerem inference (analogicznie do `bridge/whisper.js`, ale ONNX zamiast subprocess) + nowe endpointy dodane do dispatcher-a w `bridge/server.js` (wzorzec przy linii ~1247 i ~1296-1351).
- **Plik modelu (ONNX)**: poza paczką .app, w `~/Library/Application Support/Gzowo AI/` (tak jak dziś sekrety/certy — patrz `desktop/main.js`, `userDataPath`/`userData` pattern) — bridge ładuje go stamtąd przy starcie. Model NIE wchodzi do `desktop/package.json` `extraResources` whitelisty razem z kodem.
- **Historia edycji**: nowy IndexedDB store wzorowany 1:1 na `js/model3d/store.js` (`gzowo-models` → analogicznie `gzowo-photo-edits`, rekordy `{id, prompt, before(dataURL), after(dataURL), createdAt}`). To **świadomy wyjątek** od zasady "klatki Vision nigdy nie trafiają na dysk" — dotyczy WYŁĄCZNIE finalnego zdjęcia przed/po tej jednej funkcji, reszta trybu Vision (co widzisz, OCR, itd.) zostaje efemeryczna jak dziś.
- **Wyświetlanie wyniku**: w tym samym oknie asystenta, jako panel/widget obok rozmowy (spójne z resztą UI, styl Gzowo Aperture — mono, `.glass` gdzie pasuje).

## 2a. Architektura v2: cross-attention zamiast FiLM (2026-07-22)

Po pierwszym treningu (FiLM, step 8000/20k par) test na realnym zdjęciu ("add a hat") pokazał, że model **nie potrafi lokalizować edycji obiektowych** — FiLM warunkuje globalnie (jedna skala/przesunięcie na cały obraz z pojedynczego, spłaszczonego wektora tekstu), więc nie ma mechanizmu "to słowo → to miejsce w obrazie". Jurek zdecydował: **chce dodawania obiektów**, więc to wymaga realnej zmiany architektury, nie tylko więcej danych.

Zmiana:
- `model/clip_encoder.py`: zamiast pojedynczego pooled wektora (`CLIPTextModelWithProjection.text_embeds`), zwraca **pełną sekwencję per-token** (`CLIPTextModel.last_hidden_state`, stały pad do `SEQ_LEN=32`).
- `model/unet.py`: **cross-attention** do tej sekwencji po każdym poziomie rozdzielczości (down/up) + bottleneck, zamiast FiLM na tekst. Timestep (dyfuzja) zostaje przy FiLM — to faktycznie globalna wielkość ("ile szumu usunąć"), różnica jest tylko w warunkowaniu na TEKST. Zmierzone: **14,5M parametrów** (z 12,9M), forward/DDIM/eksport ONNX zweryfikowane lokalnie.
- `data/reencode_text.py` (nowy): przelicza TYLKO embeddingi tekstu na istniejącym, już pobranym datasecie (obrazy nie muszą być pobierane drugi raz) — patrz `kaggle/03-reencode-text.py`.
- **Stary checkpoint (FiLM) niekompatybilny** — nie da się go wznowić, trening od zera na tym samym datasecie 60k. Jurek świadomie zatrzymał trwającą sesję FiLM (step ~8400/28000) żeby nie marnować limitu Kaggle na coś, co i tak trafi do kosza.
- `kaggle/02-train.py`: `STEPS` zresetowane do **8000** (nowy pomiar throughput na cross-attention, model droższy obliczeniowo niż FiLM per krok).

## 4. Co NIE wchodzi w ten etap

- iMessage / wysyłanie zdjęcia (osobna runda po core).
- Automatyczna kontrola jakości wyników (heurystyki odrzucające szum/czarny obraz).
- Łańcuchowe edycje (edycja na edycji).
- Upscaling ponad Lanczos (decyzja po pierwszych wynikach treningu).

## 5. Otwarte pytania do rozstrzygnięcia PO pierwszych wynikach treningu

- Upscaling: Lanczos vs gotowy model super-res.
- Czy 64px czy 128px daje sensowną jakość przy tej skali datasetu — wybór finalnej rozdzielczości.
- Dokładna liczba kroków odszumiania (trade-off jakość vs czas inference na CPU/MPS).
- Finalny slogan po ustaleniu liczby parametrów modelu.

## Kolejne kroki

1. ✅ Jurek zatwierdził specę.
2. ✅ `Design/gedit-{mark,mark-inverse,icon}.svg` gotowe.
3. ✅ Data/training pipeline napisany i przetestowany (`data/`, `train/`, `kaggle/`).
4. **W toku** — trening: step 8000/8000 na 20k parach zrobiony, wizualnie oceniony (SPEC.md decyzje wyżej), dataset skalowany do 60k, `STEPS=28000`, kolejna sesja Kaggle w toku.
5. ✅ **Eksport ONNX + integracja z bridge'em gzowo-ai — zrobione i zweryfikowane end-to-end (2026-07-22).** `runtime/export_onnx.py` eksportuje U-Net, `bridge/photo-edit.js` (job-queue, DDIM w JS, ten sam schedule co `model/scheduler.py`) + `bridge/package.json` (`onnxruntime-node`, `@huggingface/transformers`) + routing w `bridge/server.js` + whitelist w `desktop/package.json`. Model w `~/Library/Application Support/Gzowo AI/models/gedit_unet.onnx` (poza paczką, zgodnie ze spec). Test przez curl na prawdziwym zdjęciu: wynik strukturalnie identyczny do lokalnego testu Python (`runtime/edit_photo.py`) — port jest poprawny.
6. ✅ `edit_camera_photo` tool (`js/vision/vision-tools.js`) + `captureModelInput()` w `js/vision/camera.js` + widget (`js/widgets/photo-edit.js`, **funkcjonalny, jeszcze bez pełnego designu/HCI review**) + IndexedDB (`js/photo-edit/store.js`) + wpięcie w `js/main.js`.
7. Pozostało: test end-to-end w realnej aplikacji (głos → kamera → wynik), potem iMessage jako runda 2, potem docelowy design pass na widgecie.

### Lekcje z integracji (2026-07-22)

- **`onnxruntime-node` też ma sufit `darwin-x64`** — najnowsze wersje (1.27+) porzuciły binarki dla Intel Maca, dokładnie jak PyTorch. Przypięte na **1.19.2** (ostatnia z binarką x64) w `bridge/package.json`, z `overrides` wymuszającym tę samą wersję dla zagnieżdżonej zależności `@huggingface/transformers`.
- **`@xenova/transformers` (2.x) ma bug**: `Tensor.data must be a typed array` — jego `Tensor` robi `Object.assign(this, new ONNXTensor(...))`, co nie kopiuje getterów z prototypu (`.data` ginie, tylko `.cpuData` zostaje). Występował z KAŻDĄ wersją `onnxruntime-node` (1.14.0 i 1.19.2). **Fix: przejście na następcę `@huggingface/transformers` (4.x)** — aktywnie rozwijany, ten sam projekt pod nową nazwą po tym jak Xenova przekazała go Hugging Face.
- Dwie oddzielne kopie `onnxruntime-node` w jednym procesie Node (nasza + zagnieżdżona w bibliotece tekstowej) dają ostrzeżenie macOS `Class ... is implemented in both ...` (ryzyko "mysterious crashes") — naprawione przez `overrides` wymuszający jedną wspólną wersję.

Powiązane: [[microg]], [[gzowo-ai]], [[stack]], [[identity]]
