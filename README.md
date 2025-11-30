# `RandomRotationFit`

`RandomRotationFit` torchvision transform. Use like `RandomRotation`, gets you a tighter crop. See the comparison below. Useful for round objects within bounding box.

<img width="1650" height="1650" alt="289348083" src="https://github.com/user-attachments/assets/b79579a4-dcf0-4369-b3b5-5dd1d6d6ab8a" />

## Side-by-side with `RandomRotation(expand=True)` on the right
<img width="1371" height="684" alt="sbs" src="https://github.com/user-attachments/assets/836bf067-25fc-41ff-8780-21c1665bb66c" />

## Side-by-side real world example
In the real world example it is easy to see how the bounding box fits much tighter without losing any information. Especially at near 45 degrees.
<img width="950" height="669" alt="sbs2" src="https://github.com/user-attachments/assets/b1131927-498b-4888-9025-820f72937cef" />

## `RandomRotation(expand=True)`
<img width="1650" height="1650" alt="289423782" src="https://github.com/user-attachments/assets/05decb21-c6fa-41e8-be38-d6eb47747380" />

## `RandomRotation(expand=False)`
<img width="1650" height="1650" alt="289348248" src="https://github.com/user-attachments/assets/e0020012-d759-4284-8cf7-43985fc7d3c4" />
