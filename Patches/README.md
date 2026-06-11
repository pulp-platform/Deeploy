# Patches

Out-of-tree fixes for files that live in the **GAP9 SDK install** (`install/gap9-sdk/...`),
which is *not* part of this git repo and therefore does **not** propagate via `git push`.
Apply these manually on any machine/board that flashes GAP9.

## board_runner.py — flasher last-chunk commit fix

**Target:** `install/gap9-sdk/utils/gapy_v2/bin/gapylib/chips/gap/gap9_v2/board_runner.py`

**Bug:** `BoardRunner.upload_from_file()` writes the SPI flash in 16 KB chunks using a
host↔chip `gap_rdy`/`host_rdy` handshake, but it only waits for chunk *N*'s write to
complete at the **top of iteration *N+1***. The **last chunk is fire-and-forget**: the
host loads it, sets `flash_run=0` + `host_rdy=1`, and exits the loop with no final wait.
gapy then loads the MRAM flasher, which **halts the chip mid-program**, so the final
16 KB chunk is never committed to flash.

**Symptom:** models whose `readfs` image extends past the last-committed chunk read
**garbage** for everything in that final chunk. For `MCUNet_cut0` at
`--defaultMemLevel=L3`, the 110 KB input pushes the weights/bias into the dead chunk →
conv computes on garbage weights → `FLT_MAX`/`NaN` output (all 36864 elements wrong).
Smaller nets (e.g. CCT) whose data fits below the boundary are unaffected — this is a
function of total `readfs` size, not model logic. GVSoC passes because its flash model
serves the whole (correct) image.

**Diagnosis:** corruption began at input file-byte 98236 = flash offset 98304 = exactly
`6 × 16384`, the start of the final 16 KB flashing chunk.

**Fix:** after the `while size > 0:` chunk loop in `upload_from_file`, wait for the chip
flasher to signal completion (it restores `flash_run` to 1 only after the last chunk's
program **and** verify finish):

```python
        # Wait for the flasher to finish erasing+programming+verifying the LAST chunk.
        while self.ocd.read(self.flash_run) != 1:
            time.sleep(0.01)
```

**Apply:** copy this `board_runner.py` over the SDK file, or hand-insert the snippet
above just before the final `print(... 100% ...)` in `upload_from_file`.
