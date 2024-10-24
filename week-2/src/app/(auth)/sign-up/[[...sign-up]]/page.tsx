import { SignUp } from "@clerk/nextjs";
import React from "react";

export default function Page() {
  return <SignUp routing="path" path="/sign-up" />;
}
