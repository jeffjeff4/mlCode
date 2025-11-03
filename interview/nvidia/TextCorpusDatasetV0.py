####This is a detailed breakdown of the design choices in TextCorpusDataset and how they solve the efficiency problems.
####
####The Core Problem
####
####We have a 100GB file.
####
####Memory Constraint: We cannot load the file into RAM.
####
####Access Constraint: We need to access any single line (document) quickly, without reading all the lines that come before it.
####
####The naive for i, line in enumerate(f) loop in __getitem__ is an "instant fail" because it's an $O(N)$ operation. To get line 50,000,000, it would have to read and discard the first 49,999,999 lines. This is unacceptably slow.
####
####Part 1: How to efficiently get __len__
####
####The __len__ method needs to return the total line count.
####
####Inefficient Method: Running sum(1 for line in open(self.file_path)) inside the __len__ method. This would re-scan the entire 100GB file every time len(dataset) is called (e.g., at the start of every training epoch).
####
####Efficient Solution (Pre-computation + Caching):
####
####Compute Once: We must accept that to get the line count, we have to read the file at least once. The key is to do this only once during the dataset's initialization.
####
####Store in Memory: In our _build_index method, as we iterate through the file to build the offset index (see Part 2), we are also counting the lines. We store this final count in self.total_lines.
####
####__len__ Method: The __len__ method simply becomes return self.total_lines. This is an $O(1)$ (instant) lookup.
####
####Cache to Disk (The "Pro" Move): Initialization is still slow the first time. To make all subsequent initializations instant, we save this count to a tiny metadata file (corpus.txt.len). The __init__ method now checks for this file. If it exists, it reads the length from there, skipping the 100GB file scan entirely.
####
####Part 2: How to efficiently get __getitem__
####
####This is the main challenge: getting line idx without reading lines 0 to idx-1.
####
####Inefficient Method: The $O(N)$ loop (for i, line in...).
####
####Efficient Solution (The Line Offset Index):
####
####The Concept: The solution is to create an "index" or "table of contents" that maps a line number (idx) to its exact byte position in the file.
####
####Building the Index: During the one-time scan (in _build_index), we use f.tell() before reading each line. f.tell() gives us the current byte offset from the start of the file. We store these offsets in a list, self.line_offsets.
####
####self.line_offsets[0] will be 0 (the start of line 0).
####
####self.line_offsets[1] will be the byte position of the start of line 1.
####
####self.line_offsets[idx] will be the byte position of the start of line idx.
####
####Caching the Index: This line_offsets list could be large (e.g., if we have 1 billion lines, 1B * 8 bytes/offset = 8GB of RAM). This is still much better than 100GB, but it's also smart to cache it to disk. We write this list of 8-byte integers to corpus.txt.index.
####
####__getitem__ Method: Now, __getitem__(idx) is incredibly fast:
####a.  Look up the offset: offset = self.line_offsets[idx] (This is an $O(1)$ list lookup).
####b.  Open the file: with open(self.file_path, 'r') as f:
####c.  Jump directly to the line: f.seek(offset) (This is an $O(1)$ disk seek).
####d.  Read the line: line = f.readline() (This is an $O(L)$ operation, where $L$ is the length of the line, not the whole file).
####e.  Return line.strip().
####
####This combination of pre-computation and disk-caching of the index is the standard, most robust way to handle this problem.
####
####Advanced Consideration: What if the Index Itself is Too Big?
####
####What if you have 50 billion lines? The 8-byte-per-line index would be 400GB, which also doesn't fit in memory!
####
####In this "ultra-large" scenario, you'd use a Sparse Index.
####
####Concept: Instead of storing the offset for every line, you store the offset for every 1000th line.
####
####Index Size: Your index is now 1000x smaller (400MB) and easily fits in memory.
####
####__getitem__(1005):
####
####Find the nearest index entry: sparse_idx = 1005 // 1000 = 1.
####
####Get the offset: offset = self.sparse_index[1] (this points to line 1000).
####
####f.seek(offset).
####
####Do a small read: for _ in range(5): line = f.readline().
####
####Trade-off: You've traded pure $O(1)$ access for a tiny bit of scanning (a maximum of 999 lines) in exchange for a tiny index. This is a common and powerful technique.



import os
import struct
import sys
from torch.utils.data import Dataset
from tqdm import tqdm  # Optional: for progress bar during indexing


class TextCorpusDataset(Dataset):
    """
    A PyTorch Dataset for handling massive text files that cannot be loaded
    into memory. It creates a byte-offset index for each line (document)
    to enable efficient O(1) access for __getitem__.

    This index is cached to disk so that subsequent initializations
    are nearly instantaneous.
    """

    def __init__(self, file_path: str, force_rebuild: bool = False):
        """
        Initializes the Dataset.

        Args:
            file_path (str): Path to the 100GB text file.
            force_rebuild (bool): If True, forces the index to be rebuilt
                                  even if it already exists.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        self.file_path = file_path
        self.index_path = file_path + ".index"
        self.len_path = file_path + ".len"

        if not force_rebuild and os.path.exists(self.index_path) and os.path.exists(self.len_path):
            print(f"Loading existing index from {self.index_path}...")
            self._load_index()
        else:
            print("Building index... (This may take a while for large files)")
            self._build_index()

    def _build_index(self):
        """
        Builds the line offset index and length file.
        Iterates through the file once, storing the byte offset of each
        new line. 'Q' is for unsigned long long (8 bytes), suitable for
        large file offsets.
        """
        self.line_offsets = []
        # 'Q' = unsigned long long (8 bytes per offset)
        offset_struct = struct.Struct('Q')

        with open(self.file_path, 'rb') as f_data, \
                open(self.index_path, 'wb') as f_index:
            offset = f_data.tell()

            # Use tqdm for a progress bar, wrapping the file iterator
            # We use os.path.getsize to estimate the number of bytes
            pbar = tqdm(total=os.path.getsize(self.file_path), unit='B', unit_scale=True, desc="Indexing")

            line = f_data.readline()
            while line:
                self.line_offsets.append(offset)
                f_index.write(offset_struct.pack(offset))

                pbar.update(len(line))
                offset = f_data.tell()
                line = f_data.readline()

            pbar.close()

        self.total_lines = len(self.line_offsets)

        with open(self.len_path, 'w') as f_len:
            f_len.write(str(self.total_lines))

        print(f"Index built with {self.total_lines} documents.")

    def _load_index(self):
        """
        Loads a pre-existing index from disk.
        """
        # 'Q' = unsigned long long (8 bytes per offset)
        offset_struct = struct.Struct('Q')

        with open(self.len_path, 'r') as f_len:
            self.total_lines = int(f_len.read())

        self.line_offsets = []
        with open(self.index_path, 'rb') as f_index:
            # Pre-allocate list for efficiency if possible, though appending
            # is also fast. For very large files, this is safer.
            # self.line_offsets = [0] * self.total_lines
            # for i in range(self.total_lines):
            #     chunk = f_index.read(offset_struct.size)
            #     self.line_offsets[i] = offset_struct.unpack(chunk)[0]

            # More pythonic way:
            while True:
                chunk = f_index.read(offset_struct.size)
                if not chunk:
                    break
                self.line_offsets.append(offset_struct.unpack(chunk)[0])

        if len(self.line_offsets) != self.total_lines:
            print("Warning: Index length mismatch. Rebuilding index.", file=sys.stderr)
            self._build_index()
        else:
            print(f"Successfully loaded index for {self.total_lines} documents.")

    def __len__(self):
        """
        Returns the total number of lines (documents) in the file.
        This is an O(1) operation as the length is pre-computed.
        """
        return self.total_lines

    def __getitem__(self, idx):
        """
        Returns the raw text string for the document at line `idx`.
        This is an O(1) seek + O(L) read operation, where L is
        the length of the line.

        We open/close the file handle here to make this class
        safe for multiprocessing (e.g., in a PyTorch DataLoader).
        """
        if idx < 0:
            idx = self.total_lines + idx

        if idx < 0 or idx >= self.total_lines:
            raise IndexError(f"Index {idx} out of range for dataset with {self.total_lines} lines.")

        try:
            # We re-open the file handle in __getitem__.
            # This is crucial for working with torch.utils.data.DataLoader
            # with num_workers > 0, as file handles cannot be
            # easily passed between processes.
            with open(self.file_path, 'r', encoding='utf-8') as f:
                # Seek to the pre-computed byte offset
                f.seek(self.line_offsets[idx])

                # Read the single line
                line = f.readline()

                # Return the cleaned string
                return line.strip()
        except Exception as e:
            print(f"Error reading file at index {idx}, offset {self.line_offsets.get(idx, 'N/A')}: {e}",
                  file=sys.stderr)
            return ""  # Return empty string on error


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Create a dummy test file
    DUMMY_FILE = "dummy_corpus.txt"
    print(f"Creating dummy file: {DUMMY_FILE}")
    with open(DUMMY_FILE, "w") as f:
        f.write("This is the first document.\n")
        f.write("Here is the second one.\n")
        f.write("\n")  # Empty lines are also documents
        f.write("This is the fourth and final document.\n")

    # 2. First initialization (will build the index)
    print("\n--- First pass (Building Index) ---")
    dataset = TextCorpusDataset(DUMMY_FILE)

    print(f"\nDataset length: {len(dataset)}")
    print(f"Item at index 0: '{dataset[0]}'")
    print(f"Item at index 1: '{dataset[1]}'")
    print(f"Item at index 2 (empty): '{dataset[2]}'")
    print(f"Item at index 3: '{dataset[3]}'")
    print(f"Item at index -1 (last): '{dataset[-1]}'")

    # 3. Second initialization (will load from index)
    print("\n--- Second pass (Loading Index) ---")
    dataset_from_cache = TextCorpusDataset(DUMMY_FILE)

    print(f"\nDataset length: {len(dataset_from_cache)}")
    print(f"Item at index 1: '{dataset_from_cache[1]}'")

    # 4. Clean up dummy files
    print("\nCleaning up...")
    os.remove(DUMMY_FILE)
    os.remove(DUMMY_FILE + ".index")
    os.remove(DUMMY_FILE + ".len")
    print("Done.")
