# HAB-P9, run 1: VOID (implementation bug)

Kept for transparency; do not use as evidence. When selection copied a winning robot's brain file,
the copy also inherited the winner's saved random seed (`PPO.load` re-seeds from the file), so
copies trained on identical random streams and stayed exact clones. The control arm collapsed to a
single genome by round 3 (all 8 robots score exactly 219.0 m), which breaks both the evolutionary
comparison and the independence assumption of the pre-registered test. Fixed by re-seeding each
robot after loading; see the amendment in docs/prereg/HABP9_evolved_drives.md.
