"use client";
import React, { useState } from "react";
import SearchBar from "@/components/SearchBar";

const HomePage = () => {
  const [query, setQuery] = useState("");

  const handleSearch = (value) => {
    setQuery(value);
    // Implement your search logic here
    console.log("Search query:", value);
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Handle image upload logic here
      console.log("Uploaded image:", file);
      // You can also use URL.createObjectURL(file) to display the image
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-black text-secondary">
      {/* <h1 className="text-3xl mb-6">Search and Upload</h1> */}
      <div className="absolute top-4 text-[clamp(24px,6vw,4px)]">
        Information Retrieval Project
      </div>
      <SearchBar
        placeholder="Search here..."
        onSearch={handleSearch}
        onImageUpload={handleImageUpload}
      />
      {/* <p className="mt-4">Current search query: {query}</p> */}
    </div>
  );
};

export default HomePage;
