// components/SearchBar.js
import React from "react";
import { GrAttachment } from "react-icons/gr";

const SearchBar = ({ placeholder, onSearch, onImageUpload }) => {
  return (
    <div className="flex items-center gap-2 w-full font-grotesk mx-auto px-4 py-2 relative max-w-4xl bg-primary rounded-full">
      {/* Image Upload Button */}
      <div className="">
        <label htmlFor="file-input" className="cursor-pointer">
          <GrAttachment size={24} />
        </label>
        <input
          type="file"
          id="file-input"
          accept="image/*"
          onChange={onImageUpload}
          className="hidden" // Hide the default file input
        />
      </div>

      {/* Search Input */}
      <input
        type="text"
        placeholder={placeholder}
        onChange={(e) => onSearch(e.target.value)}
        className="w-full p-2 rounded-r-lg bg-transparent focus:outline-none focus:ring-none"
      />
      {/* <button
        type="button"
        onClick={() =>
          onSearch(document.querySelector('input[type="text"]').value)
        }
        className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition"
      >
        Search
      </button> */}

      <div className="">
        <div className="min-w-8">
          <span className="" data-state="closed">
            <button
              aria-label="Send prompt"
              data-testid="send-button"
              className="flex h-8 w-8 items-center justify-center rounded-full transition-colors hover:opacity-70 focus-visible:outline-none focus-visible:outline-black disabled:text-[#f4f4f4] disabled:hover:opacity-100 dark:focus-visible:outline-white disabled:dark:bg-token-text-quaternary dark:disabled:text-token-main-surface-secondary bg-black text-white dark:bg-white dark:text-black bg-[#D7D7D7]"
            >
              <svg
                width="32"
                height="32"
                viewBox="0 0 32 32"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                className="icon-2xl"
              >
                <path
                  fill-rule="evenodd"
                  clip-rule="evenodd"
                  d="M15.1918 8.90615C15.6381 8.45983 16.3618 8.45983 16.8081 8.90615L21.9509 14.049C22.3972 14.4953 22.3972 15.2189 21.9509 15.6652C21.5046 16.1116 20.781 16.1116 20.3347 15.6652L17.1428 12.4734V22.2857C17.1428 22.9169 16.6311 23.4286 15.9999 23.4286C15.3688 23.4286 14.8571 22.9169 14.8571 22.2857V12.4734L11.6652 15.6652C11.2189 16.1116 10.4953 16.1116 10.049 15.6652C9.60265 15.2189 9.60265 14.4953 10.049 14.049L15.1918 8.90615Z"
                  fill="currentColor"
                ></path>
              </svg>
            </button>
          </span>
        </div>
      </div>
    </div>
  );
};

export default SearchBar;
