# The Basics of Operations

## What are Operations?

The underlying method for passing data/signals from component to component inside of
contexts is through the use of cables. A large amount of the time compartments will have a single cable being passed
connected to it that overwrites the previous value in that compartment. However, there are times when this is not the
case and then cable operations must be used.

## Built-in operations

By default, ngclearn comes with four operations defined, `overwrite`, `negate`, `summation` and `add`. Of these four
operations the default one used by all cables is the overwrite operation. This operations will take the value of its
source compartment and place it into the destination compartment overwriting the value currently there. The negate
operation has a similar effect as the overwrite operation with the added functionality of applying the `-` operation to
the value being transmitted. The summation operation takes in any number of source compartments and sums together all
their values and overwrites the previous value with the sum. Finally, the add operation does the same thing as the
summation operation but instead adds the sum to the previous value instead of overwriting it.

## Building Custom Operation

At its core, an operation is a static method that does all the runtime logic of the operation with the source
compartments, and a resolver that does clean up and assignment of the output of the operation to the destination
compartment.

> General Form of an Operation:
> ```python
> class operationName(BaseOp):
>   @staticmethod
>   def operation(*sources):
>       #Runtime Logic
>       return computed_value
> ```

> Example Operation (Summation)
> ```python
> class summation(BaseOp):
>     @staticmethod
>     def operation(*sources):
>         s = None
>         for source in sources:
>             if s is None:
>                 s = source
>             else:
>                 s += source
>         return s
> ```

## Notes

- Every cable coming into or out of a compartment can have a different operation.

- The order of these operations should be the order they are wired in, but this is not guaranteed.

- Only the logic that exists in the static method `operation` is used for a compiled operation, all logic existing in an overwritten resolve method is not captured.

- Some operations have a flag of `is_compilable` set to false. This is checked during compile to flag if the model can be compiled. 

- Operations can be nested so `summaion(negate(c1), c2)` would be a valid operation and will work while compiled

